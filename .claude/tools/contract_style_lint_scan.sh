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
can[[:space:]]+trust[[:print:]]{0,80}(authoritative|evidence|report)
evidence[[:print:]]{0,80}necessary[[:print:]]{0,80}(author|saying|works)
(current[[:space:]]+happy-path|happy-path[[:space:]]+example)[[:print:]]{0,80}passes
default[[:space:]]+settings[[:print:]]{0,80}(deemed[[:space:]]+compliant|satisfy|pass)
currently[[:space:]]+assumes?
exists[[:space:]]+to[[:print:]]{0,80}(recurrence|failure|defect)
historically
historical[[:space:]]+behavior
(xfail|skipped?[[:space:]]+tests?)[[:print:]]{0,80}(counts?[[:space:]]+as[[:space:]]+passing|marked[[:space:]]+`?met`?|acceptance)
(counts?[[:space:]]+as[[:space:]]+passing|marked[[:space:]]+`?met`?|acceptance)[[:print:]]{0,80}(xfail|skipped?[[:space:]]+tests?)
oracle[[:print:]]{0,80}(call|use|rely[[:space:]]+on)[[:print:]]{0,80}(production|helper|current[[:space:]]+fixtures?)
oracle[[:print:]]{0,80}default[[:print:]]{0,80}(close[[:space:]]+enough|sufficient)
(test|assertion)[[:print:]]{0,80}(only[[:space:]]+(shape|type)|non-crashing|does[[:space:]]+not[[:space:]]+raise)
zero[[:space:]]+rows?[[:print:]]{0,80}(evidence|count)
survivor-only[[:print:]]{0,80}sufficient
toy[[:space:]]+fixture
may[[:space:]]+add[[:space:]]+warning[[:space:]]+filters?[[:print:]]{0,80}(CI|green|checks?[[:space:]]+fail)
(catch|ignore)[[:print:]]{0,80}(TypeError|Exception|error)[[:print:]]{0,80}instead[[:space:]]+of[[:space:]]+fixing
source[[:space:]]+inspection[[:space:]]+alone
(final|native|machine)[[:print:]]{0,40}assembly[[:print:]]{0,40}is[[:space:]]+required[[:print:]]{0,80}acceptance
author[[:space:]]+discretion
prefer[[:print:]]{0,80}intent[[:print:]]{0,80}(written|rule|requirement|text)
best-effort
follow-up[[:print:]]{0,80}(fail|edge|acceptance|requirement)
(exempt[[:space:]]+from|may[[:space:]]+omit)[[:print:]]{0,40}(validation|tests?|coverage|assertions?)
without[[:space:]]+documenting
coverage[[:space:]]+is[[:space:]]+optional
missing[[:space:]]+assertion[[:print:]]{0,80}reviewer[[:print:]]{0,80}believes
advisory[[:print:]]{0,80}may[[:space:]]+not[[:space:]]+block
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
