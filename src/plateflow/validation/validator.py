import re
from dataclasses import dataclass


ALLOWED_LETTERS: frozenset[str] = frozenset("ABCDEFGHJKLMNPRSTUVYZ")

VALID_PROVINCE_CODES: frozenset[str] = frozenset(
    f"{i:02d}" for i in range(1, 82)
)

VALID_LETTER_DIGIT_COMBOS: tuple[tuple[int, int], ...] = (
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 2),
    (3, 3),
)

DIGIT_POSITION_CORRECTIONS: dict[str, str] = {
    "O": "0",
    "I": "1",
    "B": "8",
    "S": "5",
    "Z": "2",
    "G": "6",
}

LETTER_POSITION_CORRECTIONS: dict[str, str] = {
    "0": "O",
    "1": "I",
    "8": "B",
    "5": "S",
    "2": "Z",
    "6": "G",
}

PLATE_REGEX = re.compile(
    r"^(\d{2})([A-Z]{1,3})(\d{2,4})$"
)


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    normalized_text: str
    display_text: str
    confidence_penalty: float


def normalize_raw_text(raw: str) -> str:
    cleaned = re.sub(r"[\s\-_.]", "", raw)
    return cleaned.upper()


def apply_positional_corrections(text: str) -> str:
    match = PLATE_REGEX.match(text)
    if match:
        return text

    digits_only = re.sub(r"[^0-9A-Z]", "", text)
    if len(digits_only) < 5:
        return text

    province_part = ""
    for i in range(min(2, len(digits_only))):
        ch = digits_only[i]
        corrected = DIGIT_POSITION_CORRECTIONS.get(ch, ch)
        province_part += corrected

    remaining = digits_only[len(province_part):]

    letter_part = ""
    for ch in remaining:
        if ch.isalpha() or ch in LETTER_POSITION_CORRECTIONS:
            corrected = LETTER_POSITION_CORRECTIONS.get(ch, ch)
            if corrected in ALLOWED_LETTERS:
                letter_part += corrected
        else:
            break
        if len(letter_part) >= 3:
            break

    digit_part = ""
    after_letters = remaining[len(letter_part):]
    for ch in after_letters:
        if ch.isdigit() or ch in DIGIT_POSITION_CORRECTIONS:
            corrected = DIGIT_POSITION_CORRECTIONS.get(ch, ch)
            digit_part += corrected
        elif ch.isalpha():
            corrected_d = DIGIT_POSITION_CORRECTIONS.get(ch, "")
            if corrected_d:
                digit_part += corrected_d
        else:
            break

    return province_part + letter_part + digit_part


def build_display_text(province: str, letters: str, digits: str) -> str:
    return f"{province} {letters} {digits}"


def validate_plate(text: str) -> ValidationResult:
    normalized = normalize_raw_text(text)
    confidence_penalty = 0.0

    corrected = apply_positional_corrections(normalized)
    if corrected != normalized:
        confidence_penalty = 0.15

    match = PLATE_REGEX.match(corrected)
    if not match:
        return ValidationResult(
            is_valid=False,
            normalized_text=corrected,
            display_text=corrected,
            confidence_penalty=0.0,
        )

    province, letters, digits = match.group(1), match.group(2), match.group(3)

    if province not in VALID_PROVINCE_CODES:
        return ValidationResult(
            is_valid=False,
            normalized_text=corrected,
            display_text=corrected,
            confidence_penalty=0.0,
        )

    invalid_letters = set(letters) - ALLOWED_LETTERS
    if invalid_letters:
        return ValidationResult(
            is_valid=False,
            normalized_text=corrected,
            display_text=corrected,
            confidence_penalty=0.0,
        )

    combo = (len(letters), len(digits))
    if combo not in VALID_LETTER_DIGIT_COMBOS:
        return ValidationResult(
            is_valid=False,
            normalized_text=corrected,
            display_text=corrected,
            confidence_penalty=0.0,
        )

    display = build_display_text(province, letters, digits)

    return ValidationResult(
        is_valid=True,
        normalized_text=corrected,
        display_text=display,
        confidence_penalty=confidence_penalty,
    )
