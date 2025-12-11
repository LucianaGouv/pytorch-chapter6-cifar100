#!/usr/bin/env python3
import re
from pathlib import Path
import base64
import mimetypes
import sys
import urllib.request

ROOT = Path('.').resolve()
MD = ROOT / 'article' / 'medium_draft.md'
OUT1 = ROOT / 'article' / 'Cifar100_published_embedded.html'
OUT2 = ROOT / 'docs' / 'index.html'

if not MD.exists():
    print('Markdown not found:', MD)
    sys.exit(1)

text = MD.read_text(encoding='utf8')

img_pattern = re.compile(r'!\[(.*?)\]\((.*?)\)')

def find_local_file(path_str):
    p = Path(path_str)
    # try direct
    if p.is_absolute():
        if p.exists():
            return p
    else:
        cand = (ROOT / p)
        if cand.exists():
            return cand
    # try relative to article folder
    cand = (ROOT / 'article' / path_str)
    if cand.exists():
        return cand
    cand = (ROOT / 'figures' / path_str)
    if cand.exists():
        return cand
    cand = (ROOT / 'docs' / 'figures' / path_str)
    if cand.exists():
        return cand
    cand = (ROOT / 'notebooks' / path_str)
    if cand.exists():
        return cand
    # try by basename search
    name = Path(path_str).name
    matches = list(ROOT.rglob(name))
    if matches:
        return matches[0]
    return None

replacements = []

for m in img_pattern.finditer(text):
    alt = m.group(1)
    url = m.group(2)
    if url.startswith('data:'):
        continue
    img_bytes = None
    mime = None
    if url.startswith('http://') or url.startswith('https://'):
        try:
            print('Fetching remote image', url)
            resp = urllib.request.urlopen(url)
            img_bytes = resp.read()
            mime = resp.info().get_content_type()
        except Exception as e:
            print('Failed to fetch', url, e)
            continue
    else:
        f = find_local_file(url)
        if f is None:
            print('Image not found locally for', url)
            continue
        img_bytes = f.read_bytes()
        mime_guess = mimetypes.guess_type(f.name)[0]
        mime = mime_guess or 'application/octet-stream'
    # encode
    b64 = base64.b64encode(img_bytes).decode('ascii')
    data_uri = f'data:{mime};base64,{b64}'
    img_html = f'<img src="{data_uri}" alt="{alt}" />'
    replacements.append((m.group(0), img_html))

# perform replacements
for old, new in replacements:
    text = text.replace(old, new)

# convert markdown to html
try:
    import markdown
except Exception:
    print('Please install python-markdown in your environment: .venv/bin/python -m pip install markdown')
    sys.exit(2)

html_body = markdown.markdown(text, extensions=['fenced_code','tables','codehilite'])

template = f'''<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Artigo (embedded) — CIFAR-100</title>
<link href="https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.min.css" rel="stylesheet">
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; max-width:900px; margin:2rem auto; padding:0 1rem; color:#222; line-height:1.6 }}
  h1,h2,h3 {{ font-weight:700 }}
  pre {{ background:#f6f8fa; padding:1rem; overflow:auto }}
  code {{ background:#f6f8fa; padding:0.15rem 0.3rem; border-radius:4px }}
  img {{ max-width:100%; height:auto }}
  .meta {{ color:#666; font-size:0.95rem }}
  blockquote {{ color:#555; border-left:4px solid #ddd; padding:0.5rem 1rem }}
</style>
</head>
<body>
<main>
{html_body}
</main>
<footer style="margin-top:2rem;color:#666;font-size:0.9rem">Versão publicada gerada a partir de `article/medium_final.md` com imagens embutidas.</footer>
</body>
</html>
'''

OUT1.write_text(template, encoding='utf8')
OUT2.write_text(template, encoding='utf8')
print('Wrote', OUT1, 'and', OUT2)
print('Done.')
