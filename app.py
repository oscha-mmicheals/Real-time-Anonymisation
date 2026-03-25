from flask import Flask, request, render_template
from presidio import anonymize_text
from llm import generate_llm_response

app = Flask(__name__)
chat_history = []
@app.route('/', methods=['GET', 'POST'])
def index():
    user_input = ""
    anonymized = ""
    response = ""
    anonymize_enabled = True
    technique = ""

    if request.method == 'POST':
        user_input = request.form.get('text', '').strip()
        print("Raw user input:", repr(user_input))  # Debug

        anonymize_enabled = request.form.get('anonymize') == 'on'
        technique = request.form.get('technique', '')
        print("Anonymization enabled?", anonymize_enabled)
        print("Technique:", technique)

        if anonymize_enabled:
            anonymized = anonymize_text(user_input, technique)
        else:
            anonymized = user_input

        print("Anonymized input:", repr(anonymized))  # Debug

        response = generate_llm_response(anonymized)
        print("LLM Response:", repr(response))  # Debug

        chat_history.append((anonymized, response))

    return render_template('index.html', response=response, history=chat_history,technique=technique, anonymize=anonymize_enabled)
if __name__ == '__main__':
    app.run(debug=True)
