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

tmp_patterns=$(mktemp "${TMPDIR:-/tmp}/contract_style_lint_patterns.XXXXXX") || exit 2
trap 'rm -f "$tmp_patterns"' EXIT HUP INT TERM

cat >"$tmp_patterns" <<'PATTERNS'
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
fragile
unnecessary
overkill
(always|never|all|any)[[:space:][:punct:][:alnum:]_]{0,80}(oracle|reference|existing|prior|current|config|configuration|toolchain|numba|llvm|gpu|reviewer|test|assertion|requirement)
(oracle|reference|existing|prior|current|config|configuration|toolchain|numba|llvm|gpu|reviewer|test|assertion|requirement)[[:space:][:punct:][:alnum:]_]{0,80}(always|never|all|any|guarantees?|proves?)
current[[:space:]]+(config|configuration|default|repository)
the[[:space:]]+repository[[:space:]]+(currently|always|never)
the[[:space:]]+repository[[:space:]][[:print:]]{0,80}(config|configuration)[[:print:]]{0,80}(satisfies|passes|meets|holds)
(config|configuration)[[:space:]][[:print:]]{0,40}(satisfies|passes|meets|holds)[[:space:]]+(this|the|exactly)
expected[[:space:]]+to[[:space:]]+be[[:space:]]+(at[[:space:]]+least|competitive|comparable|sufficient|adequate|faster|better|fine|enough)
exists[[:space:]]+because
review[[:space:]]+of[[:space:]]+PR[[:space:]]*#?[0-9]+
recurrence[[:space:]]+of[[:space:]]+the[[:space:]]+same[[:space:]]+defect[[:space:]]+class
supersedes[[:space:]]+the[[:space:]]+Amendment
carried[[:space:]]+(from|forward)
takes?[[:space:]]+priority
value[[:space:]]+is[[:space:]][[:print:]]{0,80}not[[:space:]]+(raw|mere|merely|just)
is[[:space:]]+what[[:space:]]+makes
without[[:space:]]+(it|this|which)[[:space:],][[:print:]]{0,80}(would|could|must)
is[[:space:]]+normative
in[[:space:]]+favor[[:space:]]+of[[:space:]]+the[[:space:]]+letter
noncompliant[[:space:]]+by[[:space:]]+definition
takes[[:space:]]+precedence
satisfies[[:space:]]+all[[:space:]]+requirements
requirement[[:space:]]+set[[:space:]]+is[[:space:]]+not[[:space:]]+internally
PATTERNS

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
