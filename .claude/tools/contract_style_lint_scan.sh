#!/bin/sh
set -u

usage() {
  printf 'Usage: %s CONTRACT_FILE [CONTEXT_LINES]\n' "$0" >&2
}

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  usage
  exit 2
fi

contract_file=$1
context_lines=${2:-1}

if [ ! -f "$contract_file" ]; then
  printf 'contract_style_lint_scan: not a file: %s\n' "$contract_file" >&2
  exit 2
fi

case $context_lines in
  ''|*[!0-9]*)
    printf 'contract_style_lint_scan: CONTEXT_LINES must be a nonnegative integer\n' >&2
    exit 2
    ;;
esac

tmp_raw_patterns=$(mktemp "${TMPDIR:-/tmp}/contract_style_lint_patterns.raw.XXXXXX") || exit 2
tmp_patterns=$(mktemp "${TMPDIR:-/tmp}/contract_style_lint_patterns.XXXXXX") || exit 2
trap 'rm -f "$tmp_raw_patterns" "$tmp_patterns"' EXIT HUP INT TERM

cat >"$tmp_raw_patterns" <<'PATTERNS'
neither[[:space:]]+needs[[:space:]]+nor[[:space:]]+is[[:space:]]+prohibited
not[[:space:]]+required
not[[:space:]]+prohibited
not[[:space:]]+allowed
may[[:space:]]+not[[:space:]]+demand
no[[:space:]]+(requirement|assertion|test|reviewer|implementation)[[:space:]]+may
does[[:space:]]+not[[:space:]]+by[[:space:]]+itself[[:space:]]+fail
not[[:space:]]+an[[:space:]]+auto-?fail
must[[:space:]]+(explicitly[[:space:]]+)?surface
for[[:space:]]+human[[:space:]]+review
without[[:space:]]+(inspecting|examining|reviewing|checking|relying)
no[[:space:]]+reliance[[:space:]]+on
reviewers?[[:space:]]+(need[[:space:]]+not|must[[:space:]]+not|should[[:space:]]+not)
(fragile|unnecessary|overkill)[[:print:]]{0,60}(inspection|review|test|check|comparison|method|evidence)
(inspection|review|test|check|comparison|method|evidence)[[:print:]]{0,60}(fragile|unnecessary|overkill)
(^|[^[:alnum:]_])(always|never)[^[:alnum:]_][[:print:]]{0,80}(oracle|reference|existing|prior|current|config|configuration|toolchain|numba|llvm|gpu|reviewer|test|assertion|requirement)
(oracle|reference|existing|prior|current|config|configuration|toolchain|numba|llvm|gpu|reviewer|test|assertion|requirement)[[:print:]]{0,80}[^[:alnum:]_](always|never|guarantees?|proves?)([^[:alnum:]_]|$)
current[[:space:]]+(config|configuration|default|repository)
the[[:space:]]+repository[[:space:]]+(currently|always|never)
the[[:space:]]+repository[[:space:]][[:print:]]{0,80}(config|configuration)[[:print:]]{0,80}(satisfies|passes|meets|holds)
(config|configuration)[[:space:]][[:print:]]{0,40}(satisfies|passes|meets|holds)[[:space:]]+(this|the|exactly)
(default|current)[[:space:]]+(config|configuration)[[:print:]]{0,80}(satisfies|satisfied|met|passes|holds|comfortably)
(config|configuration)[[:print:]]{0,80}(satisfies|satisfied|met|passes|holds|comfortably)
expected[[:space:]]+to[[:space:]]+be[[:space:]]+(at[[:space:]]+least|competitive|comparable|sufficient|adequate|faster|better|fine|enough)
exists[[:space:]]+because
review[[:space:]]+of[[:space:]]+PR[[:space:]]*#?[0-9]+
recurrence[[:space:]]+of[[:space:]]+the[[:space:]]+same[[:space:]]+defect[[:space:]]+class
supersedes[[:space:]]+the[[:space:]]+Amendment
(supersedes|superseded|overrides|carried[[:space:]]+(from|forward))[[:space:]]+the
carried[[:space:]]+(from|forward)
(clarity|correctness|speed|performance)[[:print:]]{0,50}takes?[[:space:]]+priority
(value|point|goal|purpose)([[:space:]]+of[[:print:]]{0,40})?[[:space:]]+is[[:space:]][[:print:]]{0,80}not[[:space:]]+(raw|mere|merely|just)
is[[:space:]]+what[[:space:]]+makes
without[[:space:]]+(it|this|which)[[:space:],][[:print:]]{0,80}(would|could|must)
is[[:space:]]+normative
in[[:space:]]+favor[[:space:]]+of[[:space:]]+the[[:space:]]+letter
noncompliant[[:space:]]+by[[:space:]]+definition
takes[[:space:]]+precedence
satisfies[[:space:]]+all[[:space:]]+requirements
requirement[[:space:]]+set[[:space:]]+is[[:space:]]+not[[:space:]]+internally
(maintainer|author)[[:space:]]+(decided|chose|intentionally|prefers?|requires?)
we[[:space:]]+intentionally[[:space:]]+(avoid|chose|prefer|require)
intentionally[[:space:]]+avoid
the[[:space:]]+accepted[[:space:]]+approach
human[[:space:]]+decision
(inspection|review|test|check|comparison|method|evidence|acceptance|compliance)[[:print:]]{0,80}(insufficient|inadequate|sufficient|suffices?)
(insufficient|inadequate|sufficient|suffices?)[[:print:]]{0,80}(inspection|review|test|check|comparison|method|evidence|acceptance|compliance)
PR[[:space:]]*#?[0-9]+[[:print:]]{0,60}surfaced
intent[[:space:]]+(controls|governs|prevails|overrides)
should[[:print:]]{0,60}must
must[[:print:]]{0,60}should
may[[:space:]]+rely[[:print:]]{0,80}not[[:space:]]+authoritative
not[[:space:]]+authoritative[[:print:]]{0,80}may[[:space:]]+rely
can[[:space:]]+be[[:space:]]+trusted
currently[[:space:]]+assumes?
exists[[:space:]]+to[[:print:]]{0,80}(recurrence|failure|defect)
historically
PATTERNS

grep -Ev '^[[:space:]]*($|#)' "$tmp_raw_patterns" >"$tmp_patterns"

printf 'contract_style_lint_scan: candidate spans only; agent must classify each hit.\n'
printf 'contract_style_lint_scan: file=%s context=%s\n\n' "$contract_file" "$context_lines"

if command -v rg >/dev/null 2>&1; then
  if rg --ignore-case --line-number --context "$context_lines" --no-heading --file "$tmp_patterns" "$contract_file"; then
    exit 0
  fi
  status=$?
  if [ "$status" -eq 1 ]; then
    printf 'contract_style_lint_scan: no candidates found\n'
    exit 0
  fi
  exit "$status"
fi

if grep -Ein -C "$context_lines" -f "$tmp_patterns" "$contract_file"; then
  exit 0
fi
status=$?
if [ "$status" -eq 1 ]; then
  printf 'contract_style_lint_scan: no candidates found\n'
  exit 0
fi
exit "$status"
