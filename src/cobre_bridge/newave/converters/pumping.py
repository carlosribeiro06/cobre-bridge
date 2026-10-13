"""Pumping station conversion for NEWAVE cases.

Reads pumping station definitions from the LIBS ``usinas-hidreletricas.csv``
file and generates ``system/pumping_stations.json`` and
``constraints/pumping_bounds.parquet``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pyarrow as pa

from cobre_bridge.cobre import schemas as cobre_schemas
from cobre_bridge.core.diagnostics import (
    Diagnostic,
    DiagnosticTable,
    Severity,
    emit,
)
from cobre_bridge.core.errors import SourceFileError

if TYPE_CHECKING:
    from cobre_bridge.newave.case import NewaveCase
    from cobre_bridge.newave.id_map import NewaveIdMap

_LOG = logging.getLogger(__name__)

PUMPING_BOUNDS_SCHEMA = pa.schema(
    [
        pa.field("pumping_station_id", pa.int32(), nullable=False),
        pa.field("stage_id", pa.int32(), nullable=False),
        pa.field("block_id", pa.int32(), nullable=True),
        pa.field("min_m3s", pa.float64(), nullable=True),
        pa.field("max_m3s", pa.float64(), nullable=True),
    ]
)


def pumping_station_id_map(case: NewaveCase) -> dict[int, int]:
    """Map NEWAVE station codes to 0-based Cobre IDs (sorted order).

    Returns an empty dict if the bombeamento switch is off or no pumping data.
    """
    if not case.switches.bombeamento.on:
        return {}
    df = case.pumping_stations
    if df is None or df.empty:
        return {}
    codes = sorted(int(c) for c in df["codigo_estacao"])
    return {code: idx for idx, code in enumerate(codes)}


def convert_pumping_stations(
    case: NewaveCase,
    id_map: NewaveIdMap,
) -> tuple[dict, dict[int, int]]:
    """Convert pumping stations to ``pumping_stations.json`` dict.

    Returns ``(json_dict, converted_station_ids)`` where ``converted_station_ids``
    maps NEWAVE codes to Cobre IDs for successfully converted stations.

    Stations missing submercado assignment are skipped with a WARNING diagnostic.
    Invalid hydro codes raise ``SourceFileError``.
    """
    schema_url = cobre_schemas.schema_url_for("system/pumping_stations.json")
    empty_result: tuple[dict, dict[int, int]] = (
        {"$schema": schema_url, "pumping_stations": []},
        {},
    )

    if not case.switches.bombeamento.on:
        return empty_result

    df = case.pumping_stations
    if df is None or df.empty:
        return empty_result

    submercado_df = case.pumping_station_submercados
    station_to_submercado: dict[int, int] = {}
    if submercado_df is not None and not submercado_df.empty:
        for _, row in submercado_df.iterrows():
            station_to_submercado[int(row["codigo_estacao"])] = int(
                row["codigo_submercado"]
            )

    station_ids = pumping_station_id_map(case)
    horizon = case.horizon
    op_date = f"{horizon.start_year}-{horizon.start_month:02d}-01"

    stations: list[dict] = []
    converted: dict[int, int] = {}
    skipped: list[tuple[int, str]] = []

    for _, row in df.sort_values("codigo_estacao").iterrows():
        code = int(row["codigo_estacao"])
        name = str(row["nome_estacao"]).strip()
        source = int(row["codigo_usina_origem"])
        dest = int(row["codigo_usina_destino"])

        submercado = station_to_submercado.get(code)
        if submercado is None:
            skipped.append((code, name))
            continue

        try:
            source_id = id_map.hydro_id(source)
        except KeyError:
            raise SourceFileError(
                f"Pumping station {code} ({name}) references unknown source "
                f"hydro code {source}",
                path="usinas-hidreletricas.csv",
                field="codigo_usina_origem",
            ) from None

        try:
            dest_id = id_map.hydro_id(dest)
        except KeyError:
            raise SourceFileError(
                f"Pumping station {code} ({name}) references unknown destination "
                f"hydro code {dest}",
                path="usinas-hidreletricas.csv",
                field="codigo_usina_destino",
            ) from None

        try:
            bus_id = id_map.bus_id(submercado)
        except KeyError:
            raise SourceFileError(
                f"Pumping station {code} ({name}) assigned to unknown "
                f"submercado {submercado}",
                path="usinas-hidreletricas.csv",
                field="codigo_submercado",
            ) from None

        stations.append(
            {
                "id": station_ids[code],
                "name": name,
                "operational_start_date": op_date,
                "bus_id": bus_id,
                "source_hydro_id": source_id,
                "destination_hydro_id": dest_id,
                "consumption_mw_per_m3s": float(row["consumo_estacao"]),
                "flow": {
                    "min_m3s": 0.0,
                    "max_m3s": float(row["bombeamento_maximo"]),
                },
            }
        )
        converted[code] = station_ids[code]

    if skipped:
        emit(
            Diagnostic(
                code="pumping-station-missing-submercado",
                severity=Severity.WARNING,
                category="Pumped storage",
                title=f"Pumping stations skipped ({len(skipped)})",
                summary=(
                    f"Skipped {len(skipped)} pumping station(s) with no submercado "
                    "assignment in LIBS."
                ),
                table=DiagnosticTable(
                    columns=["Code", "Name"],
                    rows=[[code, name] for code, name in skipped],
                    justify=["right", "left"],
                ),
                remediation=(
                    "Add ESTACAO-BOMBEAMENTO-SUBMERCADO records to the LIBS file."
                ),
            ),
            logger=_LOG,
        )

    if stations:
        emit(
            Diagnostic(
                code="pumping-stations-converted",
                severity=Severity.INFO,
                category="Pumped storage",
                title=f"Pumping stations converted ({len(stations)})",
                summary=f"Converted {len(stations)} pumping station(s).",
                table=DiagnosticTable(
                    columns=["ID", "Code", "Name", "Source", "Dest", "Bus"],
                    rows=[
                        [
                            s["id"],
                            list(converted.keys())[
                                list(converted.values()).index(s["id"])
                            ],
                            s["name"],
                            s["source_hydro_id"],
                            s["destination_hydro_id"],
                            s["bus_id"],
                        ]
                        for s in stations
                    ],
                    justify=["right", "right", "left", "right", "right", "right"],
                ),
            ),
            logger=_LOG,
        )

    return ({"$schema": schema_url, "pumping_stations": stations}, converted)


def convert_pumping_bounds(
    case: NewaveCase,
    converted_station_ids: dict[int, int],
) -> pa.Table | None:
    """Generate ``pumping_bounds.parquet`` from temporal limits.

    Returns ``None`` if no temporal limits exist, the switch is off, or no
    stations were converted.
    """
    if not case.switches.bombeamento.on:
        return None
    if not converted_station_ids:
        return None

    reader = case.usinas_hidreletricas
    if reader is None:
        return None

    try:
        df = reader.estacao_bombeamento_limites_periodo_patamar(df=True)
        if df is None or df.empty:
            return None
    except (AttributeError, KeyError, TypeError):
        return None

    horizon = case.horizon
    start_year = horizon.start_year
    start_month = horizon.start_month
    total_stages = horizon.total_stages

    rows: list[dict] = []
    for _, row in df.iterrows():
        station_code = int(row["codigo_estacao"])
        if station_code not in converted_station_ids:
            continue

        station_id = converted_station_ids[station_code]
        patamar = int(row["patamar"])
        block_id = patamar - 1  # Convert 1-based to 0-based

        dt_start = row["data_inicio"]
        dt_end = row["data_fim"]

        y, m = dt_start.year, dt_start.month
        while (y, m) <= (dt_end.year, dt_end.month):
            stage_id = (y - start_year) * 12 + (m - start_month)
            if 0 <= stage_id < total_stages:
                rows.append(
                    {
                        "pumping_station_id": station_id,
                        "stage_id": stage_id,
                        "block_id": block_id,
                        "min_m3s": float(row["limite_inferior"]),
                        "max_m3s": float(row["limite_superior"]),
                    }
                )
            m += 1
            if m > 12:
                m = 1
                y += 1

    if not rows:
        return None

    return pa.Table.from_pylist(rows, schema=PUMPING_BOUNDS_SCHEMA)
