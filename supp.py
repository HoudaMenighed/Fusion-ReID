import re

def clean_requirements(input_file="requirements.txt", output_file="requirements_clean.txt"):
    cleaned_lines = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            package = re.split(r"[<>=!~]", line)[0].strip()
            package = re.sub(r"\[.*?\]", "", package)

            cleaned_lines.append(package)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(cleaned_lines))


if __name__ == "__main__":
    clean_requirements()
