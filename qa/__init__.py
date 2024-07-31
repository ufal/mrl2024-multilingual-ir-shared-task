import os

data_folder = "data"

validation_folder = os.path.join(data_folder, "MRL_ST_2024_Val")
results_folder = os.path.join(data_folder, "results")
crafted_folder = os.path.join(data_folder, "crafted")

# symlinks, read only
collection_folder = os.path.join(data_folder, "data")
collection_mt_folder = os.path.join(data_folder, "data_mt")

# 3.0 8B
# llama3_original_path = "meta-llama/Meta-Llama-3-8B-Instruct"
# llama3_not_sharded_path = "/home/manea/personal_work_ms/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/not_sharded/llama-3-8B"

# 3.0 70B - estimated on 150GB
# llama3_original_path = "meta-llama/Meta-Llama-3-70B-Instruct"
# llama3_not_sharded_path = "/home/manea/personal_work_ms/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-70B-Instruct/not_sharded"

# 3.1 8B
llama3_original_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
llama3_not_sharded_path = "/home/manea/personal_work_ms/.cache/huggingface/hub/Meta-Llama-3.1-8B-Instruct/not_sharded"


prompt_lang_mapping = {
    "EN": {
        "yes": "yes",
        "no": "no",
        "header_0": "In this context:",
        "header_1": "Having this question:",
        "header_2": "Is the answer:",
        "header_3": "Please answer with yes or no:",
        "header_4": "Which is the right answer from A, B, C and D?",

        "sys_head": "You are an assistant trained to read the following context and answer the question with one of the options A), B), C) or D).",
        "add_head": "The correct answer is:"
    },
    "ALS": {
        "yes": "ja",
        "no": "nein",
        "header_0": "In diesem Kontext:",
        "header_1": "Und dieser Frage:",
        "header_2": "Ist die Antwort:",
        "header_3": "Bitte antworten Sie mit ja oder nein:",
        "header_4": "Welches ist die richtige Antwort von A, B, C und D?",

        "sys_head": "Sie sind ein Assistent, der darauf trainiert ist, den folgenden Kontext zu lesen und die Frage mit einer der Optionen A), B), C) oder D) zu beantworten.",
        "add_head": "Die richtige Antwort ist:"
    },
    "AZ": {
        "yes": "bəli",
        "no": "yox",
        "header_0": "Bu kontekstdə:",
        "header_1": "Və bu sual:",
        "header_2": "Cavabdır:",
        "header_3": "Zəhmət olmasa bəli və ya yox cavabı verin:",
        "header_4": "A, B, C və D-dən hansı düzgün cavabdır?",

        "sys_head": "Siz aşağıdakı konteksti oxumaq və suala A), B), C) və ya D) variantlarından biri ilə cavab vermək üçün təlim keçmiş köməkçisiniz.",
        "add_head": "Düzgün cavab budur:"
    },
    "IG": {
        "yes": "ee",
        "no": "mba",
        "header_0": "Inwe ọnọdụ a:",
        "header_1": "Ma ajụjụ a:",
        "header_2": "Ọ bụ azịza ya:",
        "header_3": "Biko zaa ee ma ọ bụ mba:",
        "header_4": "Kedu azịza ziri ezi sitere na A, B, C na D?",

        "sys_head": "Ị bụ onye inyeaka a zụrụ azụ ịgụ ihe ndị a ma jiri otu nhọrọ A), B), C) ma ọ bụ D zaa ajụjụ ahụ.",
        "add_head": "Azịza ziri ezi bụ:"
    },
    "TR": {
        "yes": "evet",
        "no": "hayir",
        "header_0": "Bu bağlama sahip olmak:",
        "header_1": "Ve bu soru:",
        "header_2": "Cevap:",
        "header_3": "Lütfen evet veya hayır şeklinde cevap verin:",
        "header_4": "A, B, C ve D'nin doğru cevabı hangisidir?",
        
        "sys_head": "Aşağıdaki bağlamı okuyup soruyu A), B), C) veya D) seçeneklerinden biriyle yanıtlamak üzere eğitilmiş bir asistansınız.",
        "add_head": "Doğru cevap:"
    },
    "YO": {
        "yes": "beeni",
        "no": "rara",
        "header_0": "Nini ọrọ-ọrọ yii:",
        "header_1": "Ati ibeere yii:",
        "header_2": "Ṣe idahun:",
        "header_3": "Jọwọ dahun pẹlu bẹẹni tabi rara:",
        "header_4": "Kini idahun ti o tọ lati A, B, C ati D?",

        "sys_head": "Iwọ jẹ oluranlọwọ ti o kọ ẹkọ lati ka ipo atẹle ati dahun ibeere naa pẹlu ọkan ninu awọn aṣayan A), B), C) tabi D).",
        "add_head": "The correct answer is:"
    },
}
