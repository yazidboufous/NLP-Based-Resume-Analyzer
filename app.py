from flask import Flask, render_template, request
import Resume_analyzer  # import your Python script

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Retrieve text from the form
        resume_text = request.form["resume"]
        job_description_text = request.form["job_description"]

        # Call your script's function with these texts
        results = Resume_analyzer.process_texts(resume_text, job_description_text)

        return render_template("index.html", results=results)

    return render_template("index.html", results=None)


if __name__ == "__main__":
    app.run(debug=True)
