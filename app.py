# -*- coding: utf-8 -*-
"""
PyCode Assistant — يشغّل نموذج Transformer صغير (مدرَّب من الصفر على CoNaLa)
يحوّل نية مكتوبة بالإنجليزية إلى سطر كود بايثون، ويبثّها توكن بتوكن.
"""

import os
import time
import numpy as np
import tensorflow as tf
from flask import Flask, Response, render_template, request, stream_with_context
from tokenizers import Tokenizer

# ---------------------------------------------------------------------------
# إعدادات ثابتة — يجب أن تطابق ما استُخدم أثناء التدريب تمامًا
# ---------------------------------------------------------------------------
MAX_LEN = 96
SPECIALS = ['<PAD>', '<UNK>', '<|nl|>', '<|code|>', '<EOS>']

MODEL_PATH = os.environ.get('MODEL_PATH', 'conala_codegen_best.keras')
TOKENIZER_PATH = os.environ.get('TOKENIZER_PATH', 'conala_bpe_tokenizer.json')

print('جارِ تحميل التوكنايزر...')
tok = Tokenizer.from_file(TOKENIZER_PATH)

print('جارِ تحميل النموذج...')
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

PAD, UNK, NL_ID, CODE_ID, EOS = (tok.token_to_id(s) for s in SPECIALS)
print('تم تحميل النموذج والتوكنايزر بنجاح.')

app = Flask(__name__)


def generate_code_stream(intent: str, max_new_tokens: int = 50,
                          temperature: float = 0.6, top_p: float = 0.9):
    """يبثّ كود بايثون تدريجيًا، توكنًا بعد توكن (top-p sampling)."""
    prompt = f'<|nl|> {intent} <|code|>'
    ids = tok.encode(prompt).ids[-(MAX_LEN - 1):]
    out_ids = []
    prev_text = ''

    for _ in range(max_new_tokens):
        x = np.full((1, MAX_LEN - 1), PAD, dtype='int32')
        x[0, :len(ids)] = ids
        logits = model(x, training=False).numpy()[0, len(ids) - 1].astype('float64')
        logits = logits / max(temperature, 1e-6)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()

        if 0 < top_p < 1:
            order = np.argsort(probs)[::-1]
            cumulative = np.cumsum(probs[order])
            k = max(1, int(np.searchsorted(cumulative, top_p) + 1))
            keep = order[:k]
            filtered = np.zeros_like(probs)
            filtered[keep] = probs[keep]
            probs = filtered / filtered.sum()

        nid = int(np.random.choice(len(probs), p=probs))
        if nid in (EOS, PAD, NL_ID, CODE_ID):
            break

        ids.append(nid)
        out_ids.append(nid)
        if len(ids) >= MAX_LEN - 1:
            break

        full_text = tok.decode(out_ids)
        delta = full_text[len(prev_text):]
        prev_text = full_text
        if delta:
            yield delta
            time.sleep(0.02)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json(force=True) or {}
    intent = (data.get('question') or '').strip()

    if not intent:
        return Response('الرجاء كتابة وصف للكود المطلوب.', mimetype='text/plain')

    def event_stream():
        try:
            for chunk in generate_code_stream(intent):
                yield chunk
        except Exception as e:
            yield f'\n# [حدث خطأ أثناء توليد الكود: {e}]'

    return Response(stream_with_context(event_stream()), mimetype='text/plain')


@app.route('/health')
def health():
    # نقطة فحص بسيطة تفيد بعض خدمات الاستضافة للتأكد أن الخادم حيّ
    return {'status': 'ok'}


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
