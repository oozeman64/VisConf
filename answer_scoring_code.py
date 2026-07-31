from decimal import Decimal, ROUND_HALF_UP, InvalidOperation


def extract_boxed_answer(text: str) -> str | None:
    matches = list(re.finditer(r"\\boxed\{", text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    pos = start
    while pos < len(text) and depth > 0:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1
    if depth == 0:
        return text[start : pos - 1].strip()
    return None


def strip_latex_units(text: Any) -> str:
    """Remove LaTeX text wrappers and spacing macros so numeric answers carrying
    units (e.g. "400 \\text{ mL}", "$400$") reduce to comparable content."""
    text = str(text)
    text = re.sub(r"\\(?:text|mathrm|mbox|operatorname)\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\left|\\right|\\,|\\;|\\!|\\ ", " ", text)
    text = text.replace("$", "")
    return text.strip()


def extract_number_str(text: Any) -> str | None:
    """Return the first numeric substring after stripping LaTeX/units, else None.

    NOTE: this is a permissive *first-number* extractor used only for diagnostics.
    Scoring uses to_quantity(), which refuses to reduce symbolic expressions.
    """
    cleaned = strip_latex_units(str(text).lower()).replace(",", "")
    match = re.search(r"[-+]?\d*\.?\d+", cleaned)
    return match.group() if match else None


_FRAC_RE = re.compile(r"\\[dt]?frac\s*\{\s*(-?\d*\.?\d+)\s*\}\s*\{\s*(-?\d*\.?\d+)\s*\}")
_RATIO_RE = re.compile(r"(-?\d*\.?\d+)\s*/\s*(-?\d*\.?\d+)")


def _pre_clean_numeric(text: Any) -> str:
    """Strip value-neutral wrappers (units, $, %, degree, commas, leading label=)."""
    t = strip_latex_units(str(text).lower())
    t = t.replace("$", "").replace(",", "")
    t = t.replace("\\%", "").replace("%", "")          # percent sign ("47.6\%" -> "47.6")
    t = re.sub(r"\^\{?\\circ\}?|\\circ|\\degree|°|~", "", t)   # degree markers
    t = re.sub(r"^[a-z][a-z\s]*=", "", t).strip()      # leading "label =" prefix
    return t


def _fraction_match(cleaned: str) -> "re.Match | None":
    m = _FRAC_RE.fullmatch(cleaned) or _RATIO_RE.fullmatch(cleaned)
    if m:
        return m
    # A fraction can carry a trailing unit ("3/4 cm"); strip it the same way
    # to_quantity() strips units from plain numbers, so unit-bearing fractions
    # are recognized too (still no rounding tolerance -- see numeric_equal).
    stripped = re.sub(r"[a-z]+$", "", cleaned).strip()
    if stripped != cleaned:
        return _FRAC_RE.fullmatch(stripped) or _RATIO_RE.fullmatch(stripped)
    return None


def to_quantity(text: Any) -> "Decimal | None":
    """Return a Decimal iff text is a single numeric quantity, else None.

    Accepts a bare number optionally wrapped with units/markers that do not change
    the value: a leading "label =" (e.g. "r=7.00"), currency/percent, thousands
    commas, degree marks, and a trailing alphabetic unit (cm, mL, T, ...).

    A simple numeric fraction is EVALUATED to its value, so equivalent forms match
    (\\frac{1}{2} == 0.5 == 1/2). This is its true value, not the numerator, so
    "\\frac{22}{3}" (= 7.33) still does NOT match gold 22 -- no leniency. (Fraction
    matching is exact, no rounding -- see numeric_equal / looks_like_fraction.)

    Returns None for anything that is more than a plain number/fraction -- expressions
    with \\sqrt, \\pi, operators, or extra terms. This is deliberate: it prevents
    "2\\sqrt{3}" from being read as 2, which would falsely match gold value 2. Such
    cases fall back to strict string comparison.
    """
    t = _pre_clean_numeric(text)

    frac = _fraction_match(t)
    if frac:
        try:
            num, den = Decimal(frac.group(1)), Decimal(frac.group(2))
            return num / den if den != 0 else None
        except (InvalidOperation, ValueError, ZeroDivisionError):
            return None

    # Any LaTeX command still present (\pi, \sqrt, \cdot, \ln, ...) is value-bearing
    # -- refuse so "4 \pi" is never reduced to 4.
    if re.search(r"\\[a-zA-Z]", t):
        return None
    t = re.sub(r"[a-z]+$", "", t).strip()              # trailing alphabetic unit
    if re.fullmatch(r"[-+]?\d*\.?\d+", t):
        try:
            return Decimal(t)
        except (InvalidOperation, ValueError):
            return None
    return None


def looks_like_fraction(text: Any) -> bool:
    """True if text is a fraction form (\\frac{a}{b}, \\dfrac, \\tfrac, or a/b)."""
    return bool(_fraction_match(_pre_clean_numeric(text)))


def gold_decimals(gold: Any) -> int:
    """Number of decimal places in the gold answer string (for precision-aware match)."""
    cleaned = strip_latex_units(str(gold)).replace(",", "")
    match = re.search(r"-?\d*\.(\d+)", cleaned)
    return len(match.group(1)) if match else 0


def numeric_equal(pred: Any, gold: Any) -> bool | None:
    """Precision-aware numeric comparison, deliberately conservative.

    Returns True/False when both sides parse as numbers, else None (caller should
    fall back to string comparison).

    - If the gold has decimal places (d > 0), round the prediction to d places with
      ROUND_HALF_UP (so 1.15 -> 1.2 at 1 dp; Python's float round gives 1.1). This
      only forgives sub-precision digits the gold itself does not specify.
    - If the gold is integer-precision (d == 0), require EXACT numeric equality.
      We do NOT round a fractional prediction to the integer, so 2.4 != 2 stays
      wrong. This avoids introducing leniency relative to the old exact-match rule;
      400 == 400 (and 400.0 == 400) still pass.
    - If EITHER side is an exact fraction (\\frac{a}{b} or a/b), require exact value
      equality with no rounding tolerance, so \\frac{3}{4} (=0.75) is NOT credited
      for gold 0.8. Plain decimals still round (1.15 -> 1.2).
    """
    pd = to_quantity(pred)
    gd = to_quantity(gold)
    if pd is None or gd is None:
        return None
    if looks_like_fraction(pred) or looks_like_fraction(gold):
        return pd == gd  # exact value for fractions, no rounding
    d = gold_decimals(gold)
    if d > 0:
        quantum = Decimal(1).scaleb(-d)  # 10**-d, e.g. d=1 -> 0.1
        return pd.quantize(quantum, rounding=ROUND_HALF_UP) == gd.quantize(quantum, rounding=ROUND_HALF_UP)
    return pd == gd  # integer-precision gold: exact numeric equality only


def normalize_answer(answer: Any) -> str:
    if answer is None:
        return ""
    text = strip_latex_units(str(answer).strip().lower())
    text = re.sub(r"[$%,]", "", text)
    try:
        return f"{float(text):g}"
    except Exception:
        return text.strip()


def normalize_choice_text(text: Any) -> str:
    value = str(text or "").strip().lower()
    replacements = {
        "$": "",
        "\\(": "",
        "\\)": "",
        "\\[": "",
        "\\]": "",
        "\\left": "",
        "\\right": "",
        "\\mathrm": "",
        "\\text": "",
        "\\dfrac": "\\frac",
        "°": "^\\circ",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    value = re.sub(r"\\boxed\{(.+?)\}", r"\1", value)
    value = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", value)
    value = re.sub(r"[{}\s,，。.;；:：]", "", value)
    value = value.strip("()[]")
    return value


def parse_mcq_choices(question: str, max_choices: int = 8) -> dict[str, str]:
    # Anchor option markers to the START of a line so prose like "...point A. The
    # ..." is not mis-parsed as choice A. Real choices in these datasets are listed
    # one per line ("A. text" / "A: text" / "Choices:\nA: ...").
    letters = "".join(chr(65 + idx) for idx in range(max_choices))
    matches = list(re.finditer(rf"^\s*[\(\[]?\s*([{letters}])\s*[\.\)\]\．、:]\s*", question, re.MULTILINE))
    choices: dict[str, str] = {}
    for idx, match in enumerate(matches):
        letter = match.group(1).upper()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(question)
        choices[letter] = question[start:end].strip()
    return choices


def _last_sentence(text: str) -> str:
    """Return the last sentence of text, restricted to the final non-empty line.

    Splits on newlines first (how reasoning models structure conclusions), then
    splits within that line on '. ' + uppercase only — avoids breaking on decimal
    points like 5.25.
    """
    lines = [l.strip() for l in text.rstrip().split("\n") if l.strip()]
    line = lines[-1] if lines else text
    parts = [p.strip() for p in re.split(r"(?<![0-9])\.\s+(?=[A-Z])|[!?。]\s*", line) if p.strip()]
    return parts[-1] if parts else line


def _last_line(text: str) -> str:
    """Return the last non-empty line of text, stripped."""
    lines = [l.strip() for l in str(text).rstrip().split("\n") if l.strip()]
    return lines[-1] if lines else str(text).strip()


# Malformed boxed markup the strict \boxed{...} extractor misses: an XML-style
# "<boxed>C" tag or a backslash-less "boxed{D}". Anchored to a leading option
# letter so only genuine answer markup matches.
_BOXED_LETTER_RE = re.compile(r"(?:<boxed>|(?<![A-Za-z])\\?boxed\s*\{)\s*[\(\[]?\s*([A-Ha-h])\b")


def _boxed_letter_lenient(text: str) -> str:
    """Recover an option letter from malformed boxed markup ('<boxed>C',
    '<boxed>\\nC.', 'boxed{D}'). Returns the LAST such occurrence, or ''."""
    matches = _BOXED_LETTER_RE.findall(text)
    return matches[-1].upper() if matches else ""


def score_generation(
    generated_text: str,
    question: str,
    ground_truth: str,
    answer_type: str = "multiple_choice",
) -> dict[str, Any]:
    raw_final = extract_boxed_answer(generated_text) or ""
    gold = str(ground_truth or "").strip().upper()
    extracted = ""
    method = "none"

    if answer_type == "multiple_choice":
        # Strip LaTeX wrappers (\text{C}) then take a leading option letter that is
        # followed by a word boundary: handles "C", "(A)", "A.", "B) ...", "C: 13m",
        # "A:130°". Requires the boundary so "ANSWER" / "Apple" don't yield "A".
        # Range A-H covers up to 8 options (questions here go up to F) without
        # over-matching pronoun/article letters (I, O, ...).
        boxed_letter = strip_latex_units(raw_final).strip().upper()
        letter_match = re.match(r"^[\(\[]?\s*([A-H])\b", boxed_letter)
        if letter_match:
            extracted = letter_match.group(1)
            method = "boxed_letter"
        elif raw_final:
            boxed_norm = normalize_choice_text(raw_final)
            for letter, choice_text in parse_mcq_choices(question).items():
                if boxed_norm and boxed_norm == normalize_choice_text(choice_text):
                    extracted = letter
                    method = "boxed_option_text"
                    break
            if not extracted:
                method = "boxed_unmapped"
        if not extracted:
            # Models often mangle the boxed markup: an XML-style "<boxed>C" tag or
            # a backslash-less "boxed{D}". Recover the letter these forms carry.
            alt = _boxed_letter_lenient(generated_text)
            if alt:
                extracted = alt
                method = "boxed_letter_lenient"
        if not extracted:
            # Explicit "answer/option is X". Allow an optional "is" AND an optional
            # ":" between the cue and the letter so "answer is: A" and "option is:\nA"
            # resolve (the old (?:is|:) accepted only one of the two markers).
            last_sentence = _last_sentence(generated_text)
            patterns = [
                r"(?:answer|choice)\s*(?:is\b\s*)?:?\s*[\(\[]?\s*([A-H])\b",
                r"\boption\s*(?:is\b\s*)?:?\s*[\(\[]?\s*([A-H])\b",
            ]
            for pat in patterns:
                match = re.search(pat, last_sentence, re.I)
                if match:
                    extracted = match.group(1).upper()
                    method = "explicit_answer_letter"
                    break
        if not extracted:
            # Last resort: the final line is *just* the chosen option, e.g. "A.",
            # "A. Artistic media", "(A)". Anchor the letter to option punctuation or
            # end-of-line so prose like "A picture..." / "Because..." is NOT matched,
            # and only accept a letter that is a real option for this question.
            last_line = _last_line(generated_text)
            tail_match = re.match(r"^[\(\[]?\s*([A-H])\s*(?:[.\):\]]|$)", last_line)
            if tail_match:
                letter = tail_match.group(1).upper()
                choices = parse_mcq_choices(question)
                if not choices or letter in choices:
                    extracted = letter
                    method = "final_line_letter"
    else:
        if raw_final:
            extracted = raw_final
            method = "boxed_freeform"
        else:
            last_sentence = _last_sentence(generated_text)
            match = re.search(
                r"(?:answer|result|value\b[^.]{0,20}?)\s+(?:is|=|:)\s*([-+]?[\d,]+(?:\.\d+)?(?:\s*/\s*[\d,]+(?:\.\d+)?)?)",
                last_sentence, re.I,
            )
            if match:
                extracted = re.sub(r"[,\s]", "", match.group(1))
                method = "explicit_answer_freeform"
            else:
                method = "none"

    if answer_type == "multiple_choice":
        # Compare canonical option letters. Normalize the gold's leading letter the
        # same way as the prediction so wrapped golds like "(B)" match extracted "B".
        gold_match = re.match(r"^[\(\[]?\s*([A-H])\b", strip_latex_units(gold))
        gold_letter = gold_match.group(1) if gold_match else gold
        is_correct = bool(extracted) and extracted.strip().upper() == gold_letter
    else:
        num_eq = numeric_equal(extracted, ground_truth) if extracted else None
        if num_eq is not None:
            # Both sides numeric: compare rounded to the gold's decimal precision.
            is_correct = num_eq
        else:
            # Non-numeric (text/list): fall back to normalized string equality.
            is_correct = bool(extracted) and normalize_answer(extracted) == normalize_answer(ground_truth)
    return {
        "raw_final_answer": raw_final,
        "extracted_answer": extracted,
        "is_correct": is_correct,
        "scorer_method": method,
    }