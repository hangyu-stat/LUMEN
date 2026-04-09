import stylecloud
import wordcloud
import json
import time
import os
from collections import Counter
from PIL import Image


class HackWordCloud(wordcloud.WordCloud):
    def __init__(self, *args, **kwargs):
        kwargs['min_font_size'] = 12
        kwargs['relative_scaling'] = 0.3

        super().__init__(*args, **kwargs)

    def process_text(self, text):

        words = text.split()

        counts = Counter(words)

        return dict(counts)


stylecloud.stylecloud.WordCloud = HackWordCloud


def generate_lung_clouds_with_translation(data_source):
    MULTIPLIER = 1000000
    translation_cache = {}

    current_dir = os.path.dirname(os.path.abspath(__file__))
    font_path_to_use = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    for category_name, word_dict in data_source.items():
        print(f"Processing: {category_name} ...")

        generated_text_list = []

        for word, auc_value in word_dict.items():
            if auc_value <= 0:
                continue
            new_word = "".join(w.capitalize() for w in word.split(" "))
            repeat_times = int(auc_value * MULTIPLIER)
            generated_text_list.extend([str(new_word)] * repeat_times)

        final_text_content = " ".join(generated_text_list)
        output_filename = f"{category_name}_lung_en.png"

        stylecloud.gen_stylecloud(
            text=final_text_content,
            icon_name='fas fa-lungs',
            palette='colorbrewer.qualitative.Dark2_8',
            background_color='white',
            output_name=output_filename,
            max_words=2000,
            custom_stopwords=[''],
            collocations=False,
            font_path=font_path_to_use,
            size=2048
        )
        img = Image.open(output_filename)
        img.save(output_filename, dpi=(300, 300))
        print(f"Generated and set to 300 DPI: {output_filename}")
        print(f"Generated: {output_filename}")


if __name__ == "__main__":
    with open("./result_images/word_lg3_black_screen/auc_sorted_dict_en.json", 'r') as f:
        all_auc_sorted_dicts = json.load(f)

    try:
        generate_lung_clouds_with_translation(all_auc_sorted_dicts)
    except NameError:
        print("Some Error.")