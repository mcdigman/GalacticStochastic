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
tmp_scan_file=$(mktemp "${TMPDIR:-/tmp}/contract_style_lint_scan.XXXXXX") || exit 2
trap 'rm -f "$tmp_raw_patterns" "$tmp_patterns" "$tmp_scan_file"' EXIT HUP INT TERM

sed -e 's/[*_`]//g' "$contract_file" >"$tmp_scan_file"

cat >"$tmp_raw_patterns" <<'PATTERNS'
neither[[:space:]]+needs[[:space:]]+nor[[:space:]]+is[[:space:]]+prohibited
not[[:space:]]+((strictly|necessarily|generally|always|by[[:space:]]+itself)[[:space:]]+)?required
not[[:space:]]+((strictly|necessarily|generally|always|by[[:space:]]+itself)[[:space:]]+)?prohibited
not[[:space:]]+allowed
need[[:space:]]+not
may[[:space:]]+not[[:space:]]+demand
no[[:space:]]+(requirement|assertion|test|reviewer|implementation)[[:space:]]+may
does[[:space:]]+not[[:space:]]+by[[:space:]]+itself[[:space:]]+fail
not[[:space:]]+an[[:space:]]+auto-?fail
must[[:space:]]+(explicitly[[:space:]]+)?surface
must[[:space:]]+be[[:space:]]+surfaced
for[[:space:]]+human[[:space:]]+review
without[[:space:]]+(inspecting|examining|reviewing|checking|relying)
no[[:space:]]+reliance[[:space:]]+on
reviewers?[[:space:]]+(need[[:space:]]+not|must[[:space:]]+not|should[[:space:]]+not|(is|are)[[:space:]]+not[[:space:]]+expected|may[[:space:]]+(accept|approve|treat|rely)|can[[:space:]]+(accept|approve|treat|skip|rely))
satisf(y|ies|ied)[[:print:]]{0,60}only[[:space:]]+if[[:print:]]{0,40}reviewers?[[:space:]]+(can|must)
(fragile|unnecessary|overkill)[[:print:]]{0,60}(inspection|review|test|check|comparison|method|evidence)
(inspection|review|test|check|comparison|method|evidence)[[:print:]]{0,60}(fragile|unnecessary|overkill)
(^|[^[:alnum:]_])(always|never)[^[:alnum:]_][[:print:]]{0,50}(pass(es|ed|ing)?|fail(s|ed|ing)?|hold(s|ing)?|satisf(y|ies|ied)|compl(y|ies|ied|iant)|correct|guaranteed|suffic(es|ed|ient)?|wrong)
(oracle|reference|existing|prior|current|config|configuration|toolchain|numba|llvm|gpu|reviewer|test|assertion|requirement)[[:print:]]{0,80}[^[:alnum:]_](guarantees?|proves?)([^[:alnum:]_]|$)
the[[:space:]]+repository[[:space:]]+(currently|always|never)
the[[:space:]]+repository[[:space:]][[:print:]]{0,80}(config|configuration)[[:print:]]{0,80}(satisfies|passes|meets|holds)
(config|configuration)[[:space:]]+(satisfies|passes|meets|holds)[[:space:]]+(this|the|exactly)
(default|current)[[:space:]]+(config|configuration)[[:print:]]{0,80}(satisfies|satisfied|met|passes|holds|comfortably)
expected[[:space:]]+to[[:space:]]+be[[:space:]]+(at[[:space:]]+least|competitive|comparable|sufficient|adequate|faster|better|fine|enough)
exists[[:space:]]+because
review[[:space:]]+of[[:space:]]+PR[[:space:]]*#?[0-9]+
recurrence[[:space:]]+of[[:space:]]+the[[:space:]]+same[[:space:]]+defect[[:space:]]+class
supersedes[[:space:]]+the[[:space:]]+Amendment
(supersed(es|ed)|overrid(es|den)|carried[[:space:]]+(from|forward))([^[:alnum:]_]|$)
carried[[:space:]]+(from|forward)
revised[[:space:]]+authority
(clarity|correctness|speed|performance)[[:print:]]{0,50}takes?[[:space:]]+priority
(value|point|goal|purpose)([[:space:]]+of[[:print:]]{0,40})?[[:space:]]+is[[:space:]][[:print:]]{0,80}not[[:space:]]+(raw|mere|merely|just)
is[[:space:]]+what[[:space:]]+makes
without[[:space:]]+(it|this|which)[[:space:],][[:print:]]{0,80}(would|could|must)
is[[:space:]]+normative
(is|are|remains?|shall[[:space:]]+be)[[:space:]]+(binding|dispositive|controlling|authoritative|final|normative)
(normative[[:space:]]+force|binding[[:space:]]+guidance)
(trumps?|outranks?|prevails?|highest-authority|highest[[:space:]]+authority|above[[:space:]]+(the[[:space:]]+)?Design[[:space:]]+Intent|source[[:space:]]+of[[:space:]]+truth)
in[[:space:]]+favor[[:space:]]+of[[:space:]]+the[[:space:]]+letter
noncompliant[[:space:]]+by[[:space:]]+definition
takes[[:space:]]+precedence
satisfies[[:space:]]+all[[:space:]]+requirements
meets[[:space:]]+every[[:space:]]+requirement
passes[[:space:]]+all[[:space:]]+gates
requirement[[:space:]]+set[[:space:]]+is[[:space:]]+not[[:space:]]+internally
(maintainer|author)[[:space:]]+(decided|chose|intentionally|prefers?|requires?)
we[[:space:]]+intentionally[[:space:]]+(avoid|chose|prefer|require)
intentionally[[:space:]]+avoid
the[[:space:]]+accepted[[:space:]]+approach
human[[:space:]]+decision
(escalat(e|ed|es|ing)|rout(e|ed|es|ing)|defer(red|s|ring)?|flag(ged|s|ging)?|surfac(e|ed|es|ing)|rais(e|ed|es|ing))[[:print:]]{0,40}(human|maintainer|reviewer)
(must|should|has[[:space:]]+to)[[:space:]]+be[[:space:]]+(surfaced|raised|flagged)
pending[[:space:]]+human
(inspection|review|test|check|comparison|method|evidence|acceptance|compliance)[[:print:]]{0,80}(insufficient|inadequate|sufficient|suffices?)
(insufficient|inadequate|sufficient|suffices?)[[:print:]]{0,80}(inspection|review|test|check|comparison|method|evidence|acceptance|compliance)
PR[[:space:]]*#?[0-9]+[[:print:]]{0,60}surfaced
added[[:space:]]+in[[:space:]]+response[[:space:]]+to[[:space:]]+PR[[:space:]]*#?[0-9]+
stems[[:space:]]+from[[:space:]]+the[[:space:]]+same[[:space:]]+defect[[:space:]]+class
intent[[:space:]]+(controls|governs|prevails|overrides)
should[[:print:]]{0,30}(but|yet|however)[[:print:]]{0,20}must
must[[:print:]]{0,30}(but|yet|however)[[:print:]]{0,20}should
may[[:space:]]+rely[[:print:]]{0,80}not[[:space:]]+authoritative
not[[:space:]]+authoritative[[:print:]]{0,80}may[[:space:]]+rely
(author|implementer|report|handoff|claim|evidence|number)[[:print:]]{0,40}can[[:space:]]+be[[:space:]]+trusted
can[[:space:]]+trust[[:print:]]{0,80}(authoritative|evidence|report)
evidence[[:print:]]{0,80}necessary[[:print:]]{0,80}(author|saying|works)
(take|taking)[[:space:]]+the[[:space:]]+implementer'?s[[:space:]]+word
face[[:space:]]+value
handoff[[:print:]]{0,40}sufficient[[:space:]]+evidence
(current[[:space:]]+happy-path|happy-path[[:space:]]+example)[[:print:]]{0,80}passes
default[[:space:]]+settings[[:print:]]{0,80}(deemed[[:space:]]+compliant|satisfy|pass)
currently[[:space:]]+assumes?
already[[:space:]]+(passes|meets|complies|satisfies)
exists[[:space:]]+to[[:print:]]{0,80}(recurrence|failure|defect)
historically[[:print:]]{0,40}(pass|passes|passed|work|works|worked|compl(y|ies|ied)|fine|never|always|green|accepted)
historical[[:space:]]+behavior
(xfail|skipped?[[:space:]]+tests?)[[:print:]]{0,80}(counts?[[:space:]]+as[[:space:]]+passing|marked[[:space:]]+`?met`?|acceptance)
(counts?[[:space:]]+as[[:space:]]+passing|marked[[:space:]]+`?met`?|acceptance)[[:print:]]{0,80}(xfail|skipped?[[:space:]]+tests?)
skipped?[[:space:]]+tests?[[:print:]]{0,80}satisf(y|ies|ied)
xfail[[:print:]]{0,80}acceptable
oracle[[:print:]]{0,80}(call|use|rely[[:space:]]+on)[[:print:]]{0,80}(production|helper|current[[:space:]]+fixtures?)
oracle[[:print:]]{0,80}default[[:print:]]{0,80}(close[[:space:]]+enough|sufficient)
close[[:space:]]+enough[[:space:]]+for[[:space:]]+acceptance
good[[:space:]]+enough[[:space:]]+for[[:space:]]+now
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
best-effort[[:print:]]{0,40}(accept|accepts|accepted|suffic(es|ed|ient)?|is[[:space:]]+(fine|enough|ok)|count|counts)
follow-up[[:print:]]{0,80}(fail|edge|acceptance|requirement)
(exempt[[:space:]]+from|may[[:space:]]+omit)[[:print:]]{0,40}(validation|tests?|coverage|assertions?)
without[[:space:]]+documenting[[:print:]]{0,40}(accept|acceptance|waiver|skip|skipping|omission|coverage|evidence|rationale)
coverage[[:space:]]+is[[:space:]]+optional
missing[[:space:]]+assertion[[:print:]]{0,80}reviewer[[:print:]]{0,80}believes
advisory[[:print:]]{0,80}may[[:space:]]+not[[:space:]]+block
(^|[^[:alnum:]_])(clearly|obviously|self-?evident(ly)?|plainly|trivially|needless[[:space:]]+to[[:space:]]+say|of[[:space:]]+course)([^[:alnum:]_]|$)
PATTERNS

grep -Ev '^[[:space:]]*($|#)' "$tmp_raw_patterns" >"$tmp_patterns"

printf 'contract_style_lint_scan: candidate spans only; agent must classify each hit.\n'
printf 'contract_style_lint_scan: file=%s context=%s\n\n' "$contract_file" "$context_lines"
printf 'contract_style_lint_scan: markdown formatting delimiters (*, _, `) normalized before matching; line numbers are preserved.\n\n'

if command -v rg >/dev/null 2>&1; then
  if rg --ignore-case --line-number --context "$context_lines" --no-heading --file "$tmp_patterns" "$tmp_scan_file"; then
    exit 0
  fi
  status=$?
  if [ "$status" -eq 1 ]; then
    printf 'contract_style_lint_scan: no candidates found\n'
    exit 0
  fi
  exit "$status"
fi

if grep -Ein -C "$context_lines" -f "$tmp_patterns" "$tmp_scan_file"; then
  exit 0
fi
status=$?
if [ "$status" -eq 1 ]; then
  printf 'contract_style_lint_scan: no candidates found\n'
  exit 0
fi
exit "$status"
