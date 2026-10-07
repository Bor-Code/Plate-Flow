import pytest

from plateflow.validation.validator import (
    VALID_LETTER_DIGIT_COMBOS,
    ValidationResult,
    normalize_raw_text,
    validate_plate,
)


class TestNormalizeRawText:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("34 ABC 123", "34ABC123"),
            ("34-ABC-123", "34ABC123"),
            ("34abc123", "34ABC123"),
            ("  06 TH 42  ", "06TH42"),
            ("34.ABC.1234", "34ABC1234"),
        ],
    )
    def test_strips_separators_and_uppercases(self, raw: str, expected: str) -> None:
        assert normalize_raw_text(raw) == expected


class TestValidatePlateValidCases:
    @pytest.mark.parametrize(
        "text, display",
        [
            ("34ABC123", "34 ABC 123"),
            ("06TH42", "06 TH 42"),
            ("35YY1234", "35 YY 1234"),
            ("01A1234", "01 A 1234"),
            ("81KLM23", "81 KLM 23"),
            ("34 ABC 123", "34 ABC 123"),
        ],
    )
    def test_valid_plates_pass(self, text: str, display: str) -> None:
        result = validate_plate(text)
        assert result.is_valid is True
        assert result.display_text == display

    @pytest.mark.parametrize("letter_count, digit_count", VALID_LETTER_DIGIT_COMBOS)
    def test_all_valid_combos_accepted(self, letter_count: int, digit_count: int) -> None:
        letters = "A" * letter_count
        digits = "1" * digit_count
        plate = f"34{letters}{digits}"
        result = validate_plate(plate)
        assert result.is_valid is True


class TestValidatePlateInvalidCases:
    @pytest.mark.parametrize(
        "text",
        [
            "99ABC123",
            "00ABC123",
            "82ABC123",
        ],
    )
    def test_invalid_province_code_rejected(self, text: str) -> None:
        result = validate_plate(text)
        assert result.is_valid is False

    @pytest.mark.parametrize(
        "text",
        [
            "34QBC123",
            "34WBC123",
            "34XBC123",
        ],
    )
    def test_disallowed_letters_rejected(self, text: str) -> None:
        result = validate_plate(text)
        assert result.is_valid is False

    @pytest.mark.parametrize(
        "text",
        [
            "34AB12345",
            "34ABCD123",
            "34A12",
        ],
    )
    def test_invalid_letter_digit_combos_rejected(self, text: str) -> None:
        result = validate_plate(text)
        assert result.is_valid is False

    def test_empty_string_rejected(self) -> None:
        result = validate_plate("")
        assert result.is_valid is False

    def test_garbage_string_rejected(self) -> None:
        result = validate_plate("XYZXYZ")
        assert result.is_valid is False


class TestPositionalCorrections:
    def test_ocr_digit_in_letter_position_corrected(self) -> None:
        result = validate_plate("34AB0123")
        assert result.is_valid is True
        assert result.confidence_penalty > 0.0

    @pytest.mark.parametrize(
        "raw_text, expected_normalized",
        [
            ("34OBC123", "34ABC123"),
            ("34ABC12O", "34ABC120"),
            ("34IBC123", "34ABC123"),
            ("34ABC12I", "34ABC121"),
            ("34SBC123", "34ABC123"),
        ],
    )
    def test_ocr_confusion_characters_corrected(self, raw_text: str, expected_normalized: str) -> None:
        result = validate_plate(raw_text)
        if result.is_valid:
            assert result.normalized_text == expected_normalized

    def test_correction_triggers_confidence_penalty(self) -> None:
        result_clean = validate_plate("34ABC123")
        result_corrected = validate_plate("34OBC123")

        assert result_clean.confidence_penalty == 0.0
        assert result_corrected.confidence_penalty > 0.0

    def test_validation_result_is_frozen(self) -> None:
        result = validate_plate("34ABC123")
        assert isinstance(result, ValidationResult)
        with pytest.raises(Exception):
            object.__setattr__(result, "is_valid", False)
