"""Reserved special tokens for the shared vocabulary."""

MASK_TOKEN = "[MASK]"
PAD_TOKEN = "[PAD]"
END_TOKEN = "[END]"
DELIMITER_TOKEN = "[DELIMITER]"
WIN_TOKEN = "[WIN]"
LOSS_TOKEN = "[LOSS]"

MASK_ID = 0
PAD_ID = 1
END_ID = 2
DELIMITER_ID = 3
WIN_ID = 4
LOSS_ID = 5

SPECIAL_TOKENS: dict[str, int] = {
    MASK_TOKEN: MASK_ID,
    PAD_TOKEN: PAD_ID,
    END_TOKEN: END_ID,
    DELIMITER_TOKEN: DELIMITER_ID,
    WIN_TOKEN: WIN_ID,
    LOSS_TOKEN: LOSS_ID,
}

SPECIAL_TOKEN_IDS = frozenset(SPECIAL_TOKENS.values())

# Content tokens derived from the extractor schema start here in prompt 002+.
CONTENT_TOKEN_OFFSET = 100
