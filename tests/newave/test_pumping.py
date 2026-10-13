"""Unit tests for the pumping conversion module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pyarrow as pa
import pytest

from cobre_bridge.newave.converters.pumping import (
    PUMPING_BOUNDS_SCHEMA,
    convert_pumping_bounds,
    convert_pumping_stations,
    pumping_station_id_map,
)


# ---------------------------------------------------------------------------
# pumping_station_id_map
# ---------------------------------------------------------------------------
class TestPumpingStationIdMap:
    """Tests for pumping_station_id_map()."""

    def test_empty_when_switch_off(self) -> None:
        """Returns empty dict when bombeamento switch is off."""
        case = MagicMock()
        case.switches.bombeamento.on = False
        assert pumping_station_id_map(case) == {}

    def test_empty_when_no_data(self) -> None:
        """Returns empty dict when pumping_stations is None."""
        case = MagicMock()
        case.switches.bombeamento.on = True
        case.pumping_stations = None
        assert pumping_station_id_map(case) == {}

    def test_empty_when_dataframe_empty(self) -> None:
        """Returns empty dict when pumping_stations DataFrame is empty."""
        case = MagicMock()
        case.switches.bombeamento.on = True
        case.pumping_stations = pd.DataFrame({"codigo_estacao": []})
        assert pumping_station_id_map(case) == {}

    def test_sorted_mapping(self) -> None:
        """Returns sorted 0-based mapping from station codes."""
        case = MagicMock()
        case.switches.bombeamento.on = True
        case.pumping_stations = pd.DataFrame({"codigo_estacao": [3, 1, 2]})
        result = pumping_station_id_map(case)
        assert result == {1: 0, 2: 1, 3: 2}


# ---------------------------------------------------------------------------
# convert_pumping_stations
# ---------------------------------------------------------------------------
class TestConvertPumpingStations:
    """Tests for convert_pumping_stations()."""

    def test_empty_when_switch_off(self) -> None:
        """Returns empty list when bombeamento switch is off."""
        case = MagicMock()
        case.switches.bombeamento.on = False
        id_map = MagicMock()
        result, converted = convert_pumping_stations(case, id_map)
        assert result["pumping_stations"] == []
        assert converted == {}

    def test_empty_when_no_data(self) -> None:
        """Returns empty list when pumping_stations is None."""
        case = MagicMock()
        case.switches.bombeamento.on = True
        case.pumping_stations = None
        id_map = MagicMock()
        result, converted = convert_pumping_stations(case, id_map)
        assert result["pumping_stations"] == []
        assert converted == {}

    def test_converts_valid_station(self) -> None:
        """Converts a valid pumping station with all data."""
        case = MagicMock()
        case.switches.bombeamento.on = True
        case.pumping_stations = pd.DataFrame(
            {
                "codigo_estacao": [1],
                "nome_estacao": ["STATION_A"],
                "codigo_usina_origem": [100],
                "codigo_usina_destino": [200],
                "consumo_estacao": [1.5],
                "bombeamento_maximo": [100.0],
            }
        )
        case.pumping_station_submercados = pd.DataFrame(
            {
                "codigo_estacao": [1],
                "codigo_submercado": [2],
            }
        )
        case.horizon.start_year = 2024
        case.horizon.start_month = 1

        id_map = MagicMock()
        id_map.hydro_id = lambda code: {100: 10, 200: 20}[code]
        id_map.bus_id = lambda code: {2: 5}[code]

        result, converted = convert_pumping_stations(case, id_map)

        assert len(result["pumping_stations"]) == 1
        station = result["pumping_stations"][0]
        assert station["id"] == 0
        assert station["name"] == "STATION_A"
        assert station["source_hydro_id"] == 10
        assert station["destination_hydro_id"] == 20
        assert station["bus_id"] == 5
        assert station["consumption_mw_per_m3s"] == 1.5
        assert converted == {1: 0}

    def test_skips_station_without_submercado(self) -> None:
        """Skips station when submercado is missing (emits warning)."""
        case = MagicMock()
        case.switches.bombeamento.on = True
        case.pumping_stations = pd.DataFrame(
            {
                "codigo_estacao": [1, 2],
                "nome_estacao": ["STATION_A", "STATION_B"],
                "codigo_usina_origem": [100, 101],
                "codigo_usina_destino": [200, 201],
                "consumo_estacao": [1.5, 1.6],
                "bombeamento_maximo": [100.0, 80.0],
            }
        )
        # Only station 2 has submercado
        case.pumping_station_submercados = pd.DataFrame(
            {
                "codigo_estacao": [2],
                "codigo_submercado": [3],
            }
        )
        case.horizon.start_year = 2024
        case.horizon.start_month = 1

        id_map = MagicMock()
        id_map.hydro_id = lambda code: {100: 10, 101: 11, 200: 20, 201: 21}[code]
        id_map.bus_id = lambda code: {3: 6}[code]

        result, converted = convert_pumping_stations(case, id_map)

        # Only station 2 should be converted
        assert len(result["pumping_stations"]) == 1
        assert result["pumping_stations"][0]["name"] == "STATION_B"
        # Station 2 gets ID 1 because pumping_station_id_map assigns IDs to ALL
        # stations in sorted order, but only station 2 is successfully converted
        assert converted == {2: 1}

    def test_raises_on_invalid_hydro_code(self) -> None:
        """Raises SourceFileError when hydro code not found in id_map."""
        from cobre_bridge.core.errors import SourceFileError

        case = MagicMock()
        case.switches.bombeamento.on = True
        case.pumping_stations = pd.DataFrame(
            {
                "codigo_estacao": [1],
                "nome_estacao": ["STATION_A"],
                "codigo_usina_origem": [100],
                "codigo_usina_destino": [200],
            }
        )
        case.pumping_station_submercados = pd.DataFrame(
            {
                "codigo_estacao": [1],
                "codigo_submercado": [2],
            }
        )
        case.horizon.start_year = 2024
        case.horizon.start_month = 1

        def hydro_id_lookup(code: int) -> int:
            if code == 200:
                raise KeyError(code)
            return 10

        id_map = MagicMock()
        id_map.hydro_id = hydro_id_lookup

        with pytest.raises(SourceFileError):
            convert_pumping_stations(case, id_map)


# ---------------------------------------------------------------------------
# convert_pumping_bounds
# ---------------------------------------------------------------------------
class TestConvertPumpingBounds:
    """Tests for convert_pumping_bounds()."""

    def test_none_when_switch_off(self) -> None:
        """Returns None when bombeamento switch is off."""
        case = MagicMock()
        case.switches.bombeamento.on = False
        result = convert_pumping_bounds(case, {})
        assert result is None

    def test_none_when_empty_converted(self) -> None:
        """Returns None when no stations were converted."""
        case = MagicMock()
        case.switches.bombeamento.on = True
        result = convert_pumping_bounds(case, {})
        assert result is None

    def test_none_when_no_limits_data(self) -> None:
        """Returns None when limits DataFrame is not available."""
        case = MagicMock()
        case.switches.bombeamento.on = True
        case.usinas_hidreletricas.estacao_bombeamento_limites_periodo_patamar = (
            MagicMock(return_value=None)
        )

        result = convert_pumping_bounds(case, {1: 0})
        assert result is None

    def test_converts_bounds_correctly(self) -> None:
        """Generates correct bounds table from limits data."""
        case = MagicMock()
        case.switches.bombeamento.on = True
        case.horizon.start_year = 2024
        case.horizon.start_month = 1
        case.horizon.total_stages = 60

        # Mock limits data: station 1, patamar 1, period Jan-Feb 2024
        limits_df = pd.DataFrame(
            {
                "codigo_estacao": [1],
                "patamar": [1],
                "data_inicio": [pd.Timestamp("2024-01-01")],
                "data_fim": [pd.Timestamp("2024-02-28")],
                "limite_inferior": [0.0],
                "limite_superior": [100.0],
            }
        )
        case.usinas_hidreletricas.estacao_bombeamento_limites_periodo_patamar = (
            MagicMock(return_value=limits_df)
        )

        converted = {1: 0}
        result = convert_pumping_bounds(case, converted)

        assert result is not None
        assert result.schema == PUMPING_BOUNDS_SCHEMA
        assert result.num_rows == 2  # Jan and Feb

        df = result.to_pandas()
        assert list(df["pumping_station_id"]) == [0, 0]
        assert list(df["stage_id"]) == [0, 1]  # months 1,2 from start 2024-01
        assert list(df["block_id"]) == [0, 0]  # patamar 1 -> block 0
        assert list(df["min_m3s"]) == [0.0, 0.0]
        assert list(df["max_m3s"]) == [100.0, 100.0]

    def test_filters_only_converted_stations(self) -> None:
        """Only includes bounds for converted stations."""
        case = MagicMock()
        case.switches.bombeamento.on = True
        case.horizon.start_year = 2024
        case.horizon.start_month = 1
        case.horizon.total_stages = 60

        # Limits for stations 1 and 2, but only 1 was converted
        limits_df = pd.DataFrame(
            {
                "codigo_estacao": [1, 2],
                "patamar": [1, 1],
                "data_inicio": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")],
                "data_fim": [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-01-31")],
                "limite_inferior": [0.0, 0.0],
                "limite_superior": [100.0, 200.0],
            }
        )
        case.usinas_hidreletricas.estacao_bombeamento_limites_periodo_patamar = (
            MagicMock(return_value=limits_df)
        )

        converted = {1: 0}  # Only station 1 converted
        result = convert_pumping_bounds(case, converted)

        assert result is not None
        assert result.num_rows == 1
        assert result.to_pandas()["max_m3s"].iloc[0] == 100.0


class TestPumpingBoundsSchema:
    """Tests for the PUMPING_BOUNDS_SCHEMA constant."""

    def test_schema_fields(self) -> None:
        """Schema has expected fields with correct types."""
        assert PUMPING_BOUNDS_SCHEMA.field("pumping_station_id").type == pa.int32()
        assert PUMPING_BOUNDS_SCHEMA.field("stage_id").type == pa.int32()
        assert PUMPING_BOUNDS_SCHEMA.field("block_id").type == pa.int32()
        assert PUMPING_BOUNDS_SCHEMA.field("min_m3s").type == pa.float64()
        assert PUMPING_BOUNDS_SCHEMA.field("max_m3s").type == pa.float64()

    def test_nullable_fields(self) -> None:
        """Only pumping_station_id and stage_id are non-nullable."""
        assert not PUMPING_BOUNDS_SCHEMA.field("pumping_station_id").nullable
        assert not PUMPING_BOUNDS_SCHEMA.field("stage_id").nullable
        assert PUMPING_BOUNDS_SCHEMA.field("block_id").nullable
        assert PUMPING_BOUNDS_SCHEMA.field("min_m3s").nullable
        assert PUMPING_BOUNDS_SCHEMA.field("max_m3s").nullable


# ---------------------------------------------------------------------------
# Integration tests (require synthetic test deck)
# ---------------------------------------------------------------------------
SYNTHETIC_DECK = Path("/home/carlos.ribeiro/git2/NW-bombeamento-test")


@pytest.mark.skipif(
    not SYNTHETIC_DECK.exists(),
    reason="Synthetic deck not available",
)
class TestPumpingIntegration:
    """End-to-end tests using the synthetic pumping test deck."""

    def test_conversion_preserves_pumping_hydros(self, tmp_path: Path) -> None:
        """Hydro plants related to pumping are not removed as fictitious."""
        import json

        from cobre_bridge.newave.pipeline import convert_newave_case

        convert_newave_case(SYNTHETIC_DECK, tmp_path)

        with open(tmp_path / "system" / "hydros.json") as f:
            hydros = json.load(f)

        # The deck has 161 hydros (including pumping-related ones)
        assert len(hydros["hydros"]) >= 100

        # Check that the pumping stations reference valid hydros
        with open(tmp_path / "system" / "pumping_stations.json") as f:
            stations = json.load(f)

        hydro_ids = {h["id"] for h in hydros["hydros"]}
        for station in stations["pumping_stations"]:
            assert station["source_hydro_id"] in hydro_ids
            assert station["destination_hydro_id"] in hydro_ids

    def test_pumping_stations_json_created(self, tmp_path: Path) -> None:
        """pumping_stations.json is created with correct content."""
        import json

        from cobre_bridge.newave.pipeline import convert_newave_case

        convert_newave_case(SYNTHETIC_DECK, tmp_path)

        with open(tmp_path / "system" / "pumping_stations.json") as f:
            data = json.load(f)

        assert "pumping_stations" in data
        stations = data["pumping_stations"]
        assert len(stations) == 2  # PEDREIRA and TRAICAO

        # Check PEDREIRA station
        pedreira = next(s for s in stations if s["name"] == "PEDREIRA")
        assert pedreira["id"] == 0
        assert pedreira["consumption_mw_per_m3s"] == 1.5
        assert pedreira["flow"]["max_m3s"] == 100.0

        # Check TRAICAO station
        traicao = next(s for s in stations if s["name"] == "TRAICAO")
        assert traicao["id"] == 1
        assert traicao["consumption_mw_per_m3s"] == 1.2
        assert traicao["flow"]["max_m3s"] == 80.0

    def test_pumping_bounds_parquet_created(self, tmp_path: Path) -> None:
        """pumping_bounds.parquet is created with temporal limits."""
        import pyarrow.parquet as pq

        from cobre_bridge.newave.pipeline import convert_newave_case

        convert_newave_case(SYNTHETIC_DECK, tmp_path)

        path = tmp_path / "constraints" / "pumping_bounds.parquet"
        assert path.exists()

        table = pq.read_table(path)
        df = table.to_pandas()

        # Should have rows for patamares 1, 2, 3 (block_id 0, 1, 2)
        assert set(df["block_id"].unique()) == {0, 1, 2}

        # Both stations should have bounds
        assert set(df["pumping_station_id"].unique()) == {0, 1}

        # Check schema
        assert list(df.columns) == [
            "pumping_station_id",
            "stage_id",
            "block_id",
            "min_m3s",
            "max_m3s",
        ]
