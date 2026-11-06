"""Count words in the synthesis section of main.tex (between the SYNTHESIS markers), with macros expanded from counts.tex."""
import sys, re, os
tex = sys.argv[1]
s = open(tex, encoding='utf8').read()
cnt = {}
for line in open(os.path.join(os.path.dirname(tex), 'counts.tex'), encoding='utf8'):
    m = re.match(r'\\newcommand\{\\(\w+)\}\{([^}]*)\}', line)
    if m: cnt[m.group(1)] = m.group(2)
a = s.index('% SYNTHESIS-START'); b = s.index('% SYNTHESIS-END'); t = s[a + 17:b]
t = re.sub(r'\\cite\{[^}]*\}', '', t)
t = re.sub(r'\\(\w+)\{\}', lambda m: cnt.get(m.group(1), ''), t)
t = re.sub(r'\\(\w+)', lambda m: cnt.get(m.group(1), ' '), t)
t = re.sub(r'\$[^$]*\$', 'k', t).replace('{', '').replace('}', '').replace('~', ' ')
words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-.%]*", t)
paras = [p for p in re.split(r'\n\s*\n', s[a:b]) if p.strip()]
print('synthesis words:', len(words))
for p in paras:
    q = re.sub(r'\\cite\{[^}]*\}', '', p); q = re.sub(r'\\(\w+)', ' ', q)
    print('  para', len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-.%]*", q.replace('{', '').replace('}', ''))), p.strip()[:60].replace('\n', ' '))
