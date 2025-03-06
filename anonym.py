"""
# requirements:
# python3.7+
Faker
natasha

# Usage CLI:
# работает только с файлами в формате txt
# python3 anonym.py <input_file> <mode>
# python3 anonymize_simple/anonym.py ./1.txt fake
# python3 anonymize_simple/anonym.py ./1.txt

# usage python:
# from anonym import make_anonym
# text = 'Запрашиваемая сумма кредита: 1 000 000 (один миллион) рублей\nСрок кредитования: 36 месяцев\nЦель кредита: приобретение автомобиля\nДиректор: И.И.Иванов'
# print(make_anonym(text))
# >> 'Запрашиваемая сумма кредита: <SUM> рублей\nСрок кредитования: 36 месяцев\nЦель кредита: приобретение автомобиля\nДиректор: <PER><0>'
# print(make_anonym(text, 'fake'))
# >> 'Запрашиваемая сумма кредита: 5973 рублей\nСрок кредитования: 36 месяцев\nЦель кредита: приобретение автомобиля\nДиректор: Нинель Владимировна Баранова'
"""
import sys
import os
import re
from uuid import uuid4
from faker import Faker
import random

from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsNERTagger,
    Doc
)


segmenter = Segmenter()
morph_vocab = MorphVocab()

emb = NewsEmbedding()
ner_tagger = NewsNERTagger(emb)

fake = Faker('ru_RU')


#### NER NATASHA ####

def make_stem(t: str) -> str:
    return t.lower()[:4]


def make_norm(text):
    parts = text.split()
    return ' '.join(sorted(set([make_stem(t) for t in parts])))


def sim_pref(w1, w2):
    w1 = str(w1).lower().strip()
    w2 = str(w2).lower().strip()
    ma = max(len(w1), len(w2))
    mi = min(len(w1), len(w2))
    if mi == 0:
        raise ValueError('empty string is not allowed')
    m = 0
    while m <= mi:
        if w1[:m] != w2[:m]:
            break
        m += 1
    m -= 1
    sc = (m/ma + m/mi) / 2
    return sc


def filter_spans(span):
    stop_terms = ['beneficiar', 'principal', 'агрегатор', 'адрес', 'акцепт', 'арендатор', 'арендодатель', 'выгодоприобретатель', 'гарант', 'го',
                  'генеральный директор', 'грузоотправитель', 'грузополучатель', 'деньги', 'депозитарий', 'директор', 'дистрибьютор',
                  'доверенное лицо', 'договор', 'договорчик', 'договорённость', 'документ', 'должник', 'доступ', 'заказ', 'заказчик',
                  'заключение', 'застройщик', 'заявка', 'заявление', 'заёмщик', 'инвестор', 'информация',
                  'ип', 'исполнитель', 'истец', 'итого', 'клиент', 'калькуляция',
                  'кол-во', 'количество', 'комиссионер', 'комитент', 'контакт', 'контрагент', 'кредитор', 'лицензиар', 'лицензиат', 'номер',
                  'оборудование', 'обращение', 'обязательство', 'общество', 'ошибка',
                  'ответчик', 'отчет', 'оферента', 'оферта', 'партнер', 'перевозчик',
                  'персонал', 'письмо', 'подрядчик', 'положение', 'покупка', 'покупатель', 'поручитель', 'посредник', 'поставщик',
                  'правительство', 'право',
                  'правообладатель', 'правопреемник', 'предложение', 'предмет', 'предприниматель', 'предприятие', 'представитель', 'приказ',
                  'приложение', 'продавец', 'протокол', 'пункт', 'работ', 'работы', 'результат', 'результат работ',
                  'россии', 'российской', 'российской федерации',
                  'россия', 'рф', 'секция', 'сервис', 'соглашение', 'смета', 'сметный расчет', 'сметный',
                  'соинвестор', 'соисполнитель', 'соучредитель', 'спецификация',
                  'список', 'спор', 'срок', 'статья', 'сторона', 'страхователь', 'страховщик', 'субарендатор', 'субарендодатель',
                  'субподрядчик', 'субъект', 'сумма', 'счет', 'счетчик', 'счёт', 'телефон', 'товар', 'транспорт', 'указ', 'условие',
                  'услуги', 'участник', 'участник сделки', 'учредитель', 'учредительный документ', 'франчайзер', 'франчайзи', 'цена',
                  'цена договора', 'часть', 'экспедитор', 'экспертиза']

    s_text = span.text.lower().strip()

    if span.type == 'ORG' and len(s_text) < 4:
        return False

    if span.type == 'ORG' and re.match(r'^[^a-zа-яё]+$', s_text):
        return False


    for t in stop_terms:
        sc = sim_pref(t,s_text)
        # print(t, s_text, sc)
        if sc > 0.6:
            return False
    return True


def prepare_doc(text: str) -> Doc:
    doc = Doc(text)
    doc.segment(segmenter)
    # doc.tag_morph(morph_tagger)
    # doc.parse_syntax(syntax_parser)
    doc.tag_ner(ner_tagger)
    for span in doc.spans:
        span.normalize(morph_vocab)
    for span in doc.spans:
        span.norm_stem = make_norm(span.normal)
    return doc


def anonymize_text_natasha(text: str) -> str:
    doc = prepare_doc(text)
    spans = doc.spans
    spans = [s for s in spans if filter_spans(s)]
    tags = dict()
    for i, s in enumerate(spans):
        v = s.norm_stem
        if v not in tags:
            tags[v] = {'spans': [s], 'tag': s.type}
        else:
            tags[v]['spans'].append(s)

    cnts = {'PER': 0, 'LOC': 0, 'ORG': 0}
    for norm, sd in tags.items():
        if sd['tag'] in cnts:
            i = cnts[sd['tag']]
            repl = f"<{sd['tag']}><{i}>"
            cnts[sd['tag']] += 1
        else:
            repl = f"<{sd['tag']}>"
        for s in sd['spans']:
            text = text.replace(s.text, repl)
    return text


#### REGEX ####

def generate_num_s(n: int) -> str:
    if n <= 0:
        return ""
    return ''.join(random.choices('0123456789', k=n))


def generate_city():
    a = fake.address()
    return a.split(',')[0].strip()


def generate_street():
    a = fake.address()
    return a.split(',')[1].strip()


def replace_phone(text: str) -> str:
    pattern = r"\b\+?(?=(?:\D*\d){10,12}\D*\b)(?:\d[\d\s\-()]{7,20}\d{2,})"
    text = re.sub(pattern, "<PHONE>", text)
    text = re.sub(r'\+\s*<PHONE>', "<PHONE>", text)
    text = re.sub(r'\s*<PHONE>', " <PHONE>", text)
    return text.strip()


def replace_email(text: str) -> str:
    pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,10}"
    text = re.sub(pattern, "<EMAIL>", text)
    text = re.sub(r'\s*<EMAIL>', " <EMAIL>", text)
    return text.strip()


def replace_snils(text: str, type_='token') -> str:
    pattern = r"\b\d{3}[-\s]?\d{3}[-\s]?\d{3} \d{2}\b"
    text = re.sub(pattern, "<SNILS>", text)
    text = re.sub(r'\s*<SNILS>', " <SNILS>", text)
    return text.strip()


def replace_inn(text: str) -> str:
    patterns = [
        r'\b(\d{15})\b',
        r'\b(\d{13})\b',
        r'\b(\d{12})\b',
        r'\b(\d{10})\b',
    ]

    for pattern in patterns:
        text = re.sub(pattern, '<INN>', text)

    text = re.sub(r'\s*<INN>', " <INN>", text)
    return text.strip()


def replace_kpp(text: str) -> str:
    pattern = r'КПП:\s*\d*'
    text = re.sub(pattern, 'КПП: <KPP>', text)
    return text.strip()


def replace_bank_account(text: str) -> str:
    patterns = [
        r'\b(\d{20})\b',
    ]
    for pattern in patterns:
        text = re.sub(pattern, '<BANK_ACCOUNT>', text)
    text = re.sub(r'\s*<BANK_ACCOUNT>', " <BANK_ACCOUNT>", text)
    return text.strip()


def replace_bik(text: str) -> str:
    patterns = [
        r'БИК\s*:?\s*\d{9}',
    ]
    for pattern in patterns:
        text = re.sub(pattern, '<BIC>', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*<BIC>', " <BIC>", text)
    return text.strip()


def replace_guid(text: str) -> str:
    # Регулярное выражение для поиска GUID
    guid_pattern = r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'
    replaced_text = re.sub(guid_pattern, '<GUID>', text)
    return replaced_text


def replace_site(text: str) -> str:
    patterns = [
        r"https?://[a-zA-Z-\./&\?\w\d]*",
        r"www\.[a-zA-Z-\./&\?\w\d]*",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "<SITE>", text)

    text = re.sub(r'\s*<SITE>', " <SITE>", text)
    return text.strip()


def replace_passport(text: str) -> str:
    passport_patterns = [
        r'\b(паспорт\w?\s*(РФ)?:?|пасп\.?\s*(РФ)?:?|паспортные\s*данные:?|паспорт\s*серия:?)\s*([№N]?\d{6}[-\s/]*(код|код подразделения)?\s*\d{4}|\d{4}[\s-]*\d{6})',
        r'паспорт.{0,20}\d{4}.{0,12}\d{6}',
        r'паспорт.{0,20}\d{6}.{0,12}\d{4}',
                        ]

    def replace_match(match):
        # Возвращаем найденное слово "паспорт" + заменяем номер на <PASSPORT>
        return re.sub(r'([№N]?\d{6}[-\s/]*(код|код подразделения)?\s*\d{4}|\d{4}[\s-]*\d{6}|\d.*\d)', '<PASSPORT>', match.group(0), flags=re.IGNORECASE)

    for passport_pattern in passport_patterns:
        # print(text)
        text = re.sub(passport_pattern, replace_match, text, flags=re.IGNORECASE | re.DOTALL)

    text = re.sub(r'\s*<PASSPORT>', " <PASSPORT>", text)

    return text.strip()


def replace_money(text: str) -> str:
    pattern = r"(\d[\s\d,\.]*)\s*(?:\([а-яА-ЯёЁ\s]*\))?\s*([$€£₽]+|руб|р\.|eur|usd)"
    text = re.sub(pattern, lambda m: '<SUM> ' +  m.group(2), text, flags=re.IGNORECASE)
    text = re.sub(r'\s*<SUM>', " <SUM>", text)
    pattern2 = r"<SUM>.{0,10}(\([а-яА-ЯёЁ\s]{1,200}?руб[а-яА-ЯёЁ\s]{1,200}?\)|\([а-яА-ЯёЁ\s]{1,200}?\)\s*руб)"
    text = re.sub(pattern2, lambda m: re.sub(r'\([а-яА-ЯёЁ\s]*?\)', '', m.group(0)), text)
    return text.strip()


def replace_date(text) -> str:

    # 4. Замена даты с указанием времени в формате ДД.ММ.ГГГГ ЧЧ:ММ
    pattern4 = r'\b\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}\b'
    text = re.sub(pattern4, '<DATETIME>', text)

    # 5. Замена даты с указанием времени в формате ГГГГ-ММ-ДДTЧЧ:ММ:СС
    pattern5 = r'\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\b'
    text = re.sub(pattern5, '<DATETIME>', text)

    # 1. Замена даты в формате ДД.ММ.ГГГГ
    pattern1 = r'\b\d{2}\.\d{2}\.\d{4}\b'

    text = re.sub(pattern1, '<DATE>', text)

    # 2. Замена даты в формате ГГГГ-ММ-ДД (ISO 8601)
    pattern2 = r'\b\d{4}-\d{2}-\d{2}\b'
    text = re.sub(pattern2, '<DATE>', text)

    # 3. Замена даты в формате ММ/ДД/ГГГГ
    pattern3 = r'\b\d{2}/\d{2}/\d{4}\b'
    text = re.sub(pattern3, '<DATE>', text)

    # 6. Замена текстовых обозначений дат (например, "сегодня", "вчера")
    pattern6 = r'\b(сегодня|вчера|завтра|позавчера|послезавтра)\b'
    text = re.sub(pattern6, '<DATE>', text, flags=re.IGNORECASE)

    # 7. Замена даты в формате ДД-ММ-ГГГГ
    pattern7 = r'\b\d{2}-\d{2}-\d{4}\b'
    text = re.sub(pattern7, '<DATE>', text)

    # 8. Замена даты в формате ДД МММ ГГГГ (текстовые месяцы)
    months = r'(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)'
    pattern8 = r'\b(.{1,2})?\d{1,2}\s*(.{1,2})?\s*<months>\s*(.{1,2})?\d{4}'.replace('<months>', months)
    text = re.sub(pattern8, '<DATE>', text, flags=re.IGNORECASE)

    # 10. Замена даты в нестандартном формате ГГГГММДД
    pattern10 = r'\b\d{8}\b'
    text = re.sub(pattern10, '<DATE>', text)

    text = re.sub(r'\s*<DATE>', " <DATE>", text)
    text = re.sub(r'\s*<DATETIME>', " <DATETIME>", text)

    return text.strip()


def replace_digit(text: str) -> str:
    pattern = r'\d+'
    text = re.sub(pattern, lambda m: f'<DIGIT><{len(m.group(0))}>', text)
    return text


def replace_street(text: str) -> str:
    patterns = [
        r"(\bул\.\s*|\bпр\.\s*|\bпр\s+|\bпр-т\s*|\bпер\.\s*|\bш\.\s*|\bнаб\.\s*)[\w-]+",
        r"(\bулиц\w?\s+|\bпроспект\w?\s+|\bпереул(ок|ке)\s+|\bшоссе\s+|\bнабережн\w{2}\s+)[\w-]+"
    ]

    for pattern in patterns:
        text = re.sub(pattern, 'ул. <STREET>', text, flags=re.IGNORECASE)

    return text.strip()


def replace_building(text: str) -> str:
    patterns = [
        r"(\bд\.\s*|\bдом\w?\s*|\bстр\w?\s*|\bстроени\w?\s*)(N\s*|номер\s*|№\s*)?\d+\w?",
    ]

    for pattern in patterns:
        text = re.sub(pattern, 'д. <BNUM>', text, flags=re.IGNORECASE)

    return text.strip()


def replace_apartment(text: str) -> str:
    # Список паттернов для номеров квартир
    patterns = [
        r"\b(кв\.|квартир\w{0,4}|аппартамент(\w{1,3})?|апт?\.|apt\.?|оф\.?)\s*([№N]?\s*)?\d+\w?(-\d+\w?)?\b",  # кв. 123, квартира 45, ап. 67, apt. 89
        r"\b(к\.|k\.|к)\s*\d+\w?\b",  # к. 123, k. 45, к 67
        r"\bappart\s*\d+\w?\b",  # appart 123
        r"\bapartment\s*\d+\w?\b",  # apartment 45
        r"\bflat\s*\d+\w?\b",  # flat 67
        r"\bкомн?(\w{1,4})?\.?\s*\d+\w?\b",  # комн. 89, комнату 101
        r"\bunit\s*\d+\w?\b"  # unit 123
        r"\bофис\s*\d+\w?\b"  # unit 123
    ]
    # "Я живу в квартире 45."
    for pattern in patterns:
        text = re.sub(pattern, 'кв. <ANUM>', text, flags=re.IGNORECASE)

    return text.strip()


def replace_city(text: str) -> str:
    # Список паттернов для различных форм написания названий населенных пунктов
    patterns = [
        # Сокращенные формы
        r"\bг\.\s*[А-ЯЁа-яё-]+",
        r"\bрп\.\s*[А-ЯЁа-яё-]+",  # рабочий поселок
        r"\bпгт\s*[А-ЯЁа-яё-]+",  # поселок городского типа
        r"\bп\.\s*[А-ЯЁа-яё-]+",  # поселок
        r"\bс\.\s*[А-ЯЁа-яё-]+",  # село
        r"\bд\.\s*[А-ЯЁа-яё-]+",  # деревня
        r"\bх\.\s*[А-ЯЁа-яё-]+",  # хутор
        r"\bст\.\s*[А-ЯЁа-яё-]+",  # станица

        # Полные формы
        r"\bпоселок\s*городского\s*типа\s*[\w-]+",
        r"\bгороде?\s*[\w-]+",
        r"\bрабоч\w{2}\s*посел\w{2}\s*[\w-]+",
        r"\bпосел(ок|ке)\s*[\w-]+",
        r"\bсел(о|е)\s*[\w-]+",
        r"\bдеревн(я|е)\s*[\w-]+",
        r"\bхуторе?\s*[\w-]+",
        r"\bстаниц(а|е)\s*[\w-]+"
    ]

    for pattern in patterns:
        text = re.sub(pattern, 'г. <CITY>', text, flags=re.IGNORECASE)

    return text.strip()


def replace_fioii(text: str) -> str:
    patterns = [
        r'([Дд]иректор\w{0,3}|[Вв]рач\w{0,3}|[Сс]отрудник\w{0,3}|[Уу]правляющ\w{0,3}|[Дд]октор\w{0,3}|[Ии]нженер\w{0,3}|[Б]ухгалтер\w{0,3}|[Кк]онсультант\w{0,3}|ИП|[Ии]ндивидуальн\w{0,4}\s*[Пп]редпринимател\w{0,3})\s*:?\s*[А-ЯЁ][а-яё]+\s*[А-ЯЁ]\.(\s*[А-ЯЁ]\.)?.?',
        r'([Дд]иректор\w{0,3}|[Вв]рач\w{0,3}|[Сс]отрудник\w{0,3}|[Уу]правляющ\w{0,3}|[Дд]октор\w{0,3}|[Ии]нженер\w{0,3}|[Б]ухгалтер\w{0,3}|[Кк]онсультант\w{0,3}|ИП|[Ии]ндивидуальн\w{0,4}\s*[Пп]редпринимател\w{0,3})\s*:?\s*([А-ЯЁ]\.\s*){1,2}[А-ЯЁ][а-яё]+(-[А-ЯЁ][а-яё]+)?',
        r'([Дд]иректор\w{0,3}|[Вв]рач\w{0,3}|[Сс]отрудник\w{0,3}|[Уу]правляющ\w{0,3}|[Дд]октор\w{0,3}|[Ии]нженер\w{0,3}|[Б]ухгалтер\w{0,3}|[Кк]онсультант\w{0,3}|ИП|[Ии]ндивидуальн\w{0,4}\s*[Пп]редпринимател\w{0,3})\s+([А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\s*([А-ЯЁ][а-яё]+)?)',
        r'([Дд]иректор\w{0,3}|[Вв]рач\w{0,3}|[Сс]отрудник\w{0,3}|[Уу]правляющ\w{0,3}|[Дд]октор\w{0,3}|[Ии]нженер\w{0,3}|[Б]ухгалтер\w{0,3}|[Кк]онсультант\w{0,3}|ИП|[Ии]ндивидуальн\w{0,4}\s*[Пп]редпринимател\w{0,3})\s*:?\s*[А-ЯЁ]\.(\s*[А-ЯЁ]\.)?\s*[А-ЯЁ][а-яё]+.?'
        r'([Дд]иректор\w{0,3}|[Вв]рач\w{0,3}|[Сс]отрудник\w{0,3}|[Уу]правляющ\w{0,3}|[Дд]октор\w{0,3}|[Ии]нженер\w{0,3}|[Б]ухгалтер\w{0,3}|[Кк]онсультант\w{0,3}|ИП|[Ии]ндивидуальн\w{0,4}\s*[Пп]редпринимател\w{0,3})\s*:?\s*[А-ЯЁ][а-яё]+\s*[А-ЯЁ][а-яё]+\s*([А-ЯЁ][а-яё]+)?.?',
        r'([Дд]иректор\w{0,3}|[Вв]рач\w{0,3}|[Сс]отрудник\w{0,3}|[Уу]правляющ\w{0,3}|[Дд]октор\w{0,3}|[Ии]нженер\w{0,3}|[Б]ухгалтер\w{0,3}|[Кк]онсультант\w{0,3}|ИП|[Ии]ндивидуальн\w{0,4}\s*[Пп]редпринимател\w{0,3})\s*:?\s*[А-ЯЁ][а-яё]+\s*[А-ЯЁ][а-яё]+\s*([А-ЯЁ][а-яё]+)?.?',
        r'([Дд]иректор\w{0,3}|[Вв]рач\w{0,3}|[Сс]отрудник\w{0,3}|[Уу]правляющ\w{0,3}|[Дд]октор\w{0,3}|[Ии]нженер\w{0,3}|[Б]ухгалтер\w{0,3}|[Кк]онсультант\w{0,3}|ИП|[Ии]ндивидуальн\w{0,4}\s*[Пп]редпринимател\w{0,3})\s*:?\s*[А-ЯЁ][а-яё]+',
        r'[\b\s_/-:]+[А-ЯЁ]\.\s*[А-ЯЁ]\.\s*[А-ЯЁ][а-яё]+(-[А-ЯЁ][а-яё]+)?[\b\s_/-]+',
        r'[\b\s_/-:]+[А-ЯЁ][а-яё]+(-[А-ЯЁ][а-яё]+)?\s*[А-ЯЁ]\.\s*[А-ЯЁ][\b\s_/-]+',
        r'[^а-яёА-Яё][А-ЯЁ]\.\s*[А-ЯЁ]\.\s*[А-ЯЁ][а-яё]+(-[А-ЯЁ][а-яё]+)?[^а-яёА-Яё]',
        r'[^а-яёА-Яё][А-ЯЁ][а-яё]+(-[А-ЯЁ][а-яё]+)?\s*[А-ЯЁ]\.\s*[А-ЯЁ][^а-яёА-Яё]',
        r'([^а-яёА-Яё]|\b)[А-ЯЁ]\.\s*[А-ЯЁ]\.\s*[А-ЯЁ][а-яё]+([^а-яёА-Яё]|\b)',
        r'([^а-яёА-Яё]|\b)[А-ЯЁ][а-яё]+\s*[А-ЯЁ]\.\s*[А-ЯЁ]\.([^а-яёА-Яё]|\b)',
    ]

    for pattern in patterns:
        text = re.sub(pattern, '<PER>', text, flags=re.DOTALL)

    text = re.sub(r'/[\sа-яА-ЯЁё]{1,30}(<PER><\d+>)/', "/\g<1>/", text)
    text = re.sub(r'\s*<PER>', " <PER>", text)
    return text.strip()


def replace_org(text):
    legal_entity_types = [
        "ООО", "ЗАО", "ПАО", "ОАО", "НКО", "ИП", "АО", "ГУП", "МУП", "ФГУП",
        "СП", "КФХ", "ТСЖ", "СНТ", "ДНТ", "НП", "БЮТ", "БФ", "ВО", "ГК", "ГУ",
        "ДП", "ЖСК", "ЖК", "ЖКХ", "ЖСК", "ЖКТ", "ЖСНТ", "КХ", "КООП", "КПК",
        "КПКГ", "КПХ", "МДОУ", "МКДОУ", "МКОУ", "МОУ", "МУК", "МУП", "НВК",
        "НКО", "НП", "НПО", "ОДОУ", "ОКУ", "ООШ", "ОУ", "ПК", "ПКФ", "ПМК",
        "ПО", "ПОУ", "ППК", "ППО", "ПР", "ПТУ", "РО", "РОО", "РП", "РС\(К\)",
        "РСУ", "СДК", "СДТ", "СДЮСШОР", "СК", "СНТ", "СОШ", "СП", "СПК", "СТД",
        "СУ", "ТСЖ", "ТСН", "УК", "УМВД", "УМВД России", "УМВД РФ", "УМВД по",
        "УМВД России по", "УМВД РФ по", "УП", "УПФР", "ФГБОУ", "ФГБУ", "ФГБУН",
        "ФГБУП", "ФГКУ", "ФГУ", "ФГУП", "ФГУС", "ФКУ", "ФНС", "ФСБ", "ФСГС",
        "ФСИН", "ФСК", "ФСКН", "ФСО", "ФССП", "ФСТЭК", "ФТС", "ЦРБ", "ЦРКБ",
        "ЦРМБ", "ЦТР", "ЧОП", "ЧП", "Школа", "ЭК", "ЮЛ", "Юридическое лицо"
    ]

    # Список полных форм названий юридических лиц
    full_legal_entity_forms = [
        "общество с ограниченной ответственностью",
        "закрытое акционерное общество",
        "публичное акционерное общество",
        "открытое акционерное общество",
        "некоммерческая организация",
        "индивидуальный предприниматель",
        "акционерное общество",
        "государственное унитарное предприятие",
        "муниципальное унитарное предприятие",
        "федеральное государственное унитарное предприятие",
        "товарищество собственников жилья",
        "садоводческое некоммерческое товарищество",
        "дачное некоммерческое товарищество",
        "некоммерческое партнерство",
        "потребительский кооператив",
        "производственный кооператив",
        "кооператив",
    ]

    # Комбинируем аббревиатуры и полные формы
    all_legal_entity_patterns = (
            [entity for entity in legal_entity_types] +
            [r'\b' + form + r'\b' for form in full_legal_entity_forms]
    )

    # Регулярное выражение для поиска юридических лиц
    pattern = r'(' + '|'.join(all_legal_entity_patterns) + r''')\s*[«“"’'"](.{1,100}?)[»”"’'"]'''

    # Заменяем все совпадения в тексте
    text = re.sub(pattern, '<ORG>', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*<ORG>', " <ORG>", text)
    return text.strip()


#### FAKE TAGS REPLACER ####

def fake_tag(text: str) -> str:
    pattern = r'<PHONE>'
    text = re.sub(pattern, lambda m: fake.phone_number(), text)

    pattern = r'<DATE>'
    text = re.sub(pattern, lambda m: fake.date(), text)

    pattern = r'<DATETIME>'
    text = re.sub(pattern, lambda m: fake.date(), text)

    pattern = r'<EMAIL>'
    text = re.sub(pattern, lambda m: fake.email(), text)

    pattern = r'<SNILS>'
    text = re.sub(pattern, lambda m: generate_num_s(9) + ' ' + generate_num_s(2), text)

    pattern = r'<SUM>'
    text = re.sub(pattern, lambda m: generate_num_s(4), text)

    pattern = r'<DIGIT><(/d*)>'
    text = re.sub(pattern, lambda m: generate_num_s(int(m.group(1))), text)

    pattern = r'ул. <STREET>'
    text = re.sub(pattern, lambda m: generate_street(), text)

    pattern = r'г. <CITY>'
    text = re.sub(pattern, lambda m: generate_city(), text)

    pattern = r'<BNUM>'
    text = re.sub(pattern, lambda m: generate_num_s(3), text)

    pattern = r'<PASSPORT>'
    text = re.sub(pattern, lambda m: f'{generate_num_s(6)} {generate_num_s(4)}', text)

    pattern = r'<ANUM>'
    text = re.sub(pattern, lambda m: generate_num_s(3), text)

    pattern = r'<LOC><\d+>'
    elems = re.findall(pattern, text)

    repls = dict()
    for el in elems:
        if el not in repls:
            repls[el] = generate_city()

    for t, f in repls.items():
        text = text.replace(t, f)

    pattern = r'<LOC>'
    text = re.sub(pattern, lambda m: generate_city(), text)

    pattern = r'<ORG><\d+>'
    elems = re.findall(pattern, text)

    repls = dict()
    for el in elems:
        if el not in repls:
            repls[el] = fake.company()

    for t, f in repls.items():
        text = text.replace(t, f)

    pattern = r'<ORG>'
    text = re.sub(pattern, lambda m: fake.company(), text)

    pattern = r'<PER><\d+>'
    elems = re.findall(pattern, text)

    repls = dict()
    for el in elems:
        if el not in repls:
            repls[el] = fake.name()

    for t, f in repls.items():
        text = text.replace(t, f)

    pattern = r'<PER>'
    text = re.sub(pattern, lambda m: fake.name(), text)

    pattern = r'<INN>'
    text = re.sub(pattern, lambda m: generate_num_s(10), text)

    pattern = r'<BANK_ACCOUNT>'
    text = re.sub(pattern, lambda m: generate_num_s(20), text)

    pattern = r'<BIC>'
    text = re.sub(pattern, lambda m: 'БИК: ' + generate_num_s(9), text)

    pattern = r'<KPP>'
    text = re.sub(pattern, lambda m: generate_num_s(9), text)

    pattern = r'<SITE>'
    text = re.sub(pattern, lambda m: fake.uri(), text)

    pattern = r'<GUID>'
    text = re.sub(pattern, lambda m: str(uuid4()), text)

    return text


##### MAIN  function####

def make_anonym(text, type_='token', repalce_digits=False):
    allowed_types = ['token', 'fake']
    if type_ not in allowed_types:
        raise ValueError(f'type_ {type_} is not allowed. must be in {allowed_types}')

    text = anonymize_text_natasha(text)

    text = replace_org(text)
    text = replace_money(text)
    text = replace_passport(text)
    text = replace_bank_account(text)
    text = replace_bik(text)
    text = replace_inn(text)
    text = replace_date(text)
    text = replace_snils(text)
    text = replace_phone(text)
    text = replace_email(text)
    text = replace_site(text)
    text = replace_kpp(text)
    text = replace_guid(text)

    if repalce_digits:
        text = replace_digit(text)

    # print(text)
    text = replace_building(text)
    text = replace_apartment(text)
    # print(text)
    text = replace_street(text)
    text = replace_city(text)

    text = replace_fioii(text)

    if type_ == 'fake':
        text = fake_tag(text)

    return text


#### TESTS ####

def test_all():
    assert len(generate_num_s(1)) == 1
    assert len(generate_num_s(2)) == 2
    assert len(generate_num_s(10)) == 10

    # print(replace_phone("Мой телефон: +7 (999) 123-45-67, мое имя Иван"))
    print(replace_phone("Мой телефон: +7 (999) 123-45-67, мое имя Иван"))
    assert replace_phone("Мой телефон: +7 (999) 123-45-67, мое имя Иван") == "Мой телефон: <PHONE>, мое имя Иван"
    assert replace_phone("Мой телефон: 8-999-123-45-67, мое имя Иван") == "Мой телефон: <PHONE>, мое имя Иван"
    assert replace_phone("Мой телефон: 89991234567, мое имя Иван") == "Мой телефон: <PHONE>, мое имя Иван"

    # Тест кейс 1: Номер с пробелами внутри
    assert replace_phone("Телефон: +7 999 123 45 67") == "Телефон: <PHONE>"

    # Тест кейс 2: Номер в конце строки
    assert replace_phone("Позвоните мне по номеру 8-999-123-45-67") == "Позвоните мне по номеру <PHONE>"

    # Тест кейс 3: Несколько номеров в тексте (должен заменить все)
    print(replace_phone("Мои телефоны: +7(999)123-45-67, 8-999-123-45-67"))
    assert replace_phone("Мои телефоны: +7(999)123-45-67, 8-999-123-45-67") == "Мои телефоны: <PHONE>, <PHONE>"

    # Тест кейс 4: Номер с различными разделителями
    assert replace_phone("Телефон: +7-999-123-45-67") == "Телефон: <PHONE>"

    # Тест кейс 5: Короткий номер (например, внутренний)
    assert replace_phone("Для связи используйте телефон 1234") == "Для связи используйте телефон 1234"
    print(replace_phone("Для связи используйте телефон 12345678"))
    assert replace_phone("Для связи используйте телефон 123456") == "Для связи используйте телефон 123456"
    assert replace_phone("Для связи используйте телефон 123456789") == "Для связи используйте телефон 123456789"
    assert replace_phone("Для связи используйте телефон 1234567890") == "Для связи используйте телефон <PHONE>"
    assert replace_phone("Для связи используйте телефон 123456789012") == "Для связи используйте телефон <PHONE>"
    print(replace_phone("Для связи используйте телефон 12345678901234"))
    assert replace_phone("Для связи используйте телефон 1234567890123") == "Для связи используйте телефон 1234567890123"

    # Тест кейс 6: Номер с кодом страны и городским форматом
    assert replace_phone("Телефон компании: +7 (495) 123-45-67") == "Телефон компании: <PHONE>"

    # Тест кейс 7: Номер внутри строки с другими числами
    print(replace_phone(
        "Закупили 10 товаров. Контактный телефон: 89991234567"))
    assert replace_phone(
        "Закупили 10 товаров. Контактный телефон: 89991234567") == "Закупили 10 товаров. Контактный телефон: <PHONE>"

    # Тест кейс 8: Номер с дополнительными символами (скобки, тире, точки)
    print(replace_phone("Телефон: (8-999) 123 45 67"))
    assert replace_phone("Телефон: (8-999) 123 45 67") == "Телефон: ( <PHONE>"

    # Тест кейс 9: Международный формат номера
    assert replace_phone("Контактный телефон: +1 (202) 555-0133") == "Контактный телефон: <PHONE>"

    # Тест кейс 10: Сложный случай с комбинацией разных форматов
    assert replace_phone(
        "Напишите нам на email@example.com или позвоните: +7(999)123-45-67, 8-999-123-45-67") == "Напишите нам на email@example.com или позвоните: <PHONE>, <PHONE>"

    assert replace_email("Мои данные example@example.com, мое имя Иван") == "Мои данные <EMAIL>, мое имя Иван"
    assert replace_email(
        "Мои данные user.name+tag+sorting@domain.com, мое имя Иван") == "Мои данные <EMAIL>, мое имя Иван"
    assert replace_email("Мои данные user_name123@gmail.com, мое имя Иван") == "Мои данные <EMAIL>, мое имя Иван"
    assert replace_email("Мои данные info@sub.domain.org, мое имя Иван") == "Мои данные <EMAIL>, мое имя Иван"
    assert replace_email("Мои данные test.user+label@company.co.uk, мое имя Иван") == "Мои данные <EMAIL>, мое имя Иван"
    assert replace_email("Мои данные invalid-email@, мое имя Иван") != "Мои данные <EMAIL>, мое имя Иван"
    assert replace_email("Мои данные another_invalid_email, мое имя Иван") != "Мои данные <EMAIL>, мое имя Иван"

    assert replace_snils("СНИЛС 123-456-789 01, мое имя Иван") == "СНИЛС <SNILS>, мое имя Иван"
    assert replace_snils("СНИЛС: 123-456-789 01, мое имя Иван") == "СНИЛС: <SNILS>, мое имя Иван"
    # assert replace_snils("СНИЛС 123456789-01, мое имя Иван") == "СНИЛС <SNILS>, мое имя Иван"
    # assert replace_snils("СНИЛС: 12345678901, мое имя Иван") == "СНИЛС: <SNILS>, мое имя Иван"
    assert replace_snils("СНИЛС: 123 456 789 01, мое имя Иван") == "СНИЛС: <SNILS>, мое имя Иван"

    print(replace_snils("СНИЛС: 123 456 789 01, мое имя Иван", 'fake'))

    print(replace_money('Сумма: 1000 рублей'))
    assert replace_money('Сумма: 1000 слов') == 'Сумма: 1000 слов'
    assert replace_money("Сумма: 1000 рублей") == "Сумма: <SUM> рублей"
    assert replace_money('Сумма: 10.00 рублей') == 'Сумма: <SUM> рублей'
    assert replace_money('Значение: 1 000 000 руб.') == 'Значение: <SUM> руб.'
    assert replace_money('Пример: 123456789.123 £') == 'Пример: <SUM> £'
    assert replace_money('Цена: 5000') == 'Цена: 5000'

    # 1. Формат ДД.ММ.ГГГГ
    assert replace_date("Срок действия до 31.12.2025") == "Срок действия до <DATE>"

    # 2. Формат ГГГГ-ММ-ДД (ISO 8601)
    assert replace_date("Дата создания 2025-12-31") == "Дата создания <DATE>"

    # 3. Формат ММ/ДД/ГГГГ
    assert replace_date("Платеж выполнен 12/31/2025") == "Платеж выполнен <DATE>"

    # 4. Формат ДД.ММ.ГГГГ ЧЧ:ММ
    print(replace_date("Начало мероприятия 31.12.2025 18:00"))
    assert replace_date("Начало мероприятия 31.12.2025 18:00") == "Начало мероприятия <DATETIME>"

    # 5. Формат ГГГГ-ММ-ДДTЧЧ:ММ:СС
    assert replace_date("Лог записан 2025-12-31T18:00:00") == "Лог записан <DATETIME>"

    # 6. Текстовые обозначения дат
    assert replace_date("Это случится сегодня") == "Это случится <DATE>"
    assert replace_date("Это было вчера") == "Это было <DATE>"
    assert replace_date("Это будет завтра") == "Это будет <DATE>"

    # 7. Формат ДД-ММ-ГГГГ
    assert replace_date("Конец действия 31-12-2025") == "Конец действия <DATE>"

    # 8. Формат ДД МММ ГГГГ (текстовые месяцы)
    assert replace_date("Рождение 31 декабря 2025") == "Рождение <DATE>"

    # 9. Диапазон дат
    assert replace_date("Период с 01.01.2025 по 31.12.2025") == "Период с <DATE> по <DATE>"

    # 10. Нестандартный формат ГГГГММДД
    assert replace_date("Дата записи 20251231") == "Дата записи <DATE>"

    assert replace_digit("My number is 12345") == "My number is <DIGIT><5>"
    assert replace_digit("There are 7 days in a week") == "There are <DIGIT><1> days in a week"

    print(replace_street("Также на встрече присутствовали внешние эксперты"))
    assert replace_street("Я живу на ул. Ленина") == "Я живу на ул. <STREET>"
    print(replace_street("Офис находится на проспекте Мира"))
    assert replace_street("Офис находится на проспекте Мира") == "Офис находится на ул. <STREET>"
    print(replace_street("Переулок Садовый в центре города"))
    assert replace_street("Переулок Садовый в центре города") == "ул. <STREET> в центре города"
    assert replace_street("По шоссе Варшавское движение плотное") == "По ул. <STREET> движение плотное"
    print(replace_street("Прогулялся по набережной Невы"))
    assert replace_street("Прогулялся по набережной Невы") == "Прогулялся по ул. <STREET>"
    assert replace_street("ул. Победы красиво освещена") == "ул. <STREET> красиво освещена"
    assert replace_street("пр-т Энергетиков очень широкий") == "ул. <STREET> очень широкий"
    assert replace_street("пер. Зелёный тихий и уютный") == "ул. <STREET> тихий и уютный"
    assert replace_street("ш. Кутузовское одно из главных в Москве") == "ул. <STREET> одно из главных в Москве"
    assert replace_street("наб. Фонтанки историческая часть города") == "ул. <STREET> историческая часть города"
    assert replace_street("Улица Горького всегда оживлённая") == "ул. <STREET> всегда оживлённая"
    assert replace_street("Проспект Дружбы соединяет два района") == "ул. <STREET> соединяет два района"
    assert replace_street("Переулок Винный известен своими ресторанами") == "ул. <STREET> известен своими ресторанами"
    assert replace_street("Шоссе Рублёвское часто упоминается в песнях") == "ул. <STREET> часто упоминается в песнях"
    assert replace_street("Набережная Макарова имеет вид на крепость") == "ул. <STREET> имеет вид на крепость"
    assert replace_street("ул. Бакунина пересекается с ул. Ломоносова") == "ул. <STREET> пересекается с ул. <STREET>"
    assert replace_street("пр-т Ленина украшен фонтанами") == "ул. <STREET> украшен фонтанами"
    assert replace_street(
        "пер. Тихий находится в старой части города") == "ул. <STREET> находится в старой части города"
    assert replace_street("ш. Каширское загружено транспортом") == "ул. <STREET> загружено транспортом"
    assert replace_street("наб. Карповки популярна среди туристов") == "ул. <STREET> популярна среди туристов"
    assert replace_street("На улице Советской проходит ежегодный парад") == "На ул. <STREET> проходит ежегодный парад"

    assert replace_building("ул. Пушкина, д. 12") == "ул. Пушкина, д. <BNUM>"
    assert replace_building("дом 5А в центре города") == "д. <BNUM> в центре города"
    assert replace_building("ДОМ 7Б является историческим памятником") == "д. <BNUM> является историческим памятником"
    assert replace_building("адрес: д. 8, корпус 1") == "адрес: д. <BNUM>, корпус 1"
    assert replace_building(
        "Здание по адресу Д. 45 было построено в 1990 году") == "Здание по адресу д. <BNUM> было построено в 1990 году"
    assert replace_building("д. 3 находится рядом с парком") == "д. <BNUM> находится рядом с парком"
    assert replace_building("дом 100 - это высотное здание") == "д. <BNUM> - это высотное здание"
    assert replace_building("Дом N23 не найден в системе") == "д. <BNUM> не найден в системе"
    assert replace_building("На улице д. 6А живет много людей") == "На улице д. <BNUM> живет много людей"
    assert replace_building("Новый дом 9B был сдан в эксплуатацию") == "Новый д. <BNUM> был сдан в эксплуатацию"
    assert replace_building("Старый д. 15 требует ремонта") == "Старый д. <BNUM> требует ремонта"
    assert replace_building(
        "Указанный дом 27 является объектом охраны") == "Указанный д. <BNUM> является объектом охраны"
    assert replace_building(
        "д. 1A и д. 2B находятся в одном квартале") == "д. <BNUM> и д. <BNUM> находятся в одном квартале"
    assert replace_building("По адресу: дом 18, строение 3") == "По адресу: д. <BNUM>, д. <BNUM>"
    assert replace_building("Дом 123 находится в престижном районе") == "д. <BNUM> находится в престижном районе"
    assert replace_building("д. 11A имеет историческую ценность") == "д. <BNUM> имеет историческую ценность"
    print(replace_building("В районе дома 4B проводится реконструкция"))
    assert replace_building(
        "В районе дома 4B проводится реконструкция") == "В районе д. <BNUM> проводится реконструкция"
    assert replace_building("д. 22B был построен в 2005 году") == "д. <BNUM> был построен в 2005 году"
    assert replace_building("Дом 33А принадлежит частному лицу") == "д. <BNUM> принадлежит частному лицу"
    assert replace_building(
        "д. 7B будет отремонтирован в следующем году") == "д. <BNUM> будет отремонтирован в следующем году"
    print(replace_building("Дом номер 4C находится на окраине города"))
    assert replace_building("Дом номер 4C находится на окраине города") == "д. <BNUM> находится на окраине города"
    assert replace_building("Филиал №123, адрес: г. Москва, ул. Примерная, д. 12, тел.: +7(495)123-45-67") == \
           "Филиал №123, адрес: г. Москва, ул. Примерная, д. <BNUM>, тел.: +7(495)123-45-67"

    # Тесты с сокращенными формами
    assert replace_city("Я живу в г. Москва") == "Я живу в г. <CITY>"
    assert replace_city("Родился в рп. Иваново") == "Родился в г. <CITY>"
    assert replace_city("Учусь в пгт Зеленоград") == "Учусь в г. <CITY>"
    assert replace_city("Приехал из п. Лесной") == "Приехал из г. <CITY>"
    assert replace_city("Бабушка живет в с. Красное") == "Бабушка живет в г. <CITY>"

    # Тесты с полными формами
    assert replace_city("Город Санкт-Петербург красивый") == "г. <CITY> красивый"
    assert replace_city("В рабочем поселке Октябрьский") == "В г. <CITY>"
    print(replace_city("Поселок городского типа Новый"))
    assert replace_city("Поселок городского типа Новый") == "г. <CITY>"
    assert replace_city("Мы были в поселке Сосновый") == "Мы были в г. <CITY>"
    assert replace_city("Деревня Малая находится здесь") == "г. <CITY> находится здесь"

    # Тесты с различными окончаниями слов
    assert replace_city("В городе Москве холодно зимой") == "В г. <CITY> холодно зимой"
    assert replace_city("Станица Курганская известна своей историей") == "г. <CITY> известна своей историей"
    assert replace_city("Хутор Веселый находится рядом") == "г. <CITY> находится рядом"
    assert replace_city("Село Петровское большое") == "г. <CITY> большое"
    assert replace_city("Деревнею называют этот населенный пункт") == "г. <CITY> называют этот населенный пункт"

    # Тесты с разным регистром
    assert replace_city("Живу в Г. Москва") == "Живу в г. <CITY>"
    assert replace_city("Приехал из РП. Иваново") == "Приехал из г. <CITY>"
    assert replace_city("Учусь в ПГТ Зеленоград") == "Учусь в г. <CITY>"
    assert replace_city("Бабушка живет в С. Красное") == "Бабушка живет в г. <CITY>"
    assert replace_city("Деревня малая находится здесь") == "г. <CITY> находится здесь"

    # Тесты с несколькими упоминаниями
    assert replace_city("Я был в г. Москва и в г. Санкт-Петербург") == "Я был в г. <CITY> и в г. <CITY>"
    print(replace_city("В селе Красное и в деревне Малая"))
    assert replace_city("В селе Красное и в деревне Малая") == "В г. <CITY> и в г. <CITY>"

    # Тесты с кириллицей и латиницей
    assert replace_city("Город London да заменяется") == "г. <CITY> да заменяется"
    assert replace_city("Село Krasnoe должно замениться") == "г. <CITY> должно замениться"

    # Тесты с окончаниями на мягкий знак
    assert replace_city("Новый Поселок Лесной красивый") == "Новый г. <CITY> красивый"
    assert replace_city("старая Станица Кубанская большая") == "старая г. <CITY> большая"

    # Базовые случаи
    assert replace_passport("паспорт 349876  4595") == "паспорт <PASSPORT>"
    print(replace_passport("паспорт РФ 349876 код 4595"))
    assert replace_passport("паспорт РФ 349876 код 4595") == "паспорт РФ <PASSPORT>"
    print(replace_passport("паспорт: 349876 код подразделения 4595"))
    assert replace_passport("паспорт: 349876 код подразделения 4595") == "паспорт: <PASSPORT>"

    # Сокращения
    assert replace_passport("пасп 349876  4595") == "пасп <PASSPORT>"
    assert replace_passport("пасп. 349876  4595") == "пасп. <PASSPORT>"

    # Разные форматы записи
    assert replace_passport("Паспорт: 349876   4595") == "Паспорт: <PASSPORT>"
    assert replace_passport("Документ: паспорт 349876 4595") == "Документ: паспорт <PASSPORT>"
    assert replace_passport("Паспортные данные: 349876 4595") == "Паспортные данные: <PASSPORT>"

    # С пробелами и табуляцией
    print(replace_passport("паспорт\t349876\t4595"))
    assert replace_passport("паспорт\t349876\t4595") == "паспорт <PASSPORT>"
    assert replace_passport("паспорт  \t 349876   4595") == "паспорт <PASSPORT>"

    # В разных регистрах
    assert replace_passport("ПАСПОРТ 349876 4595") == "ПАСПОРТ <PASSPORT>"
    assert replace_passport("ПаСпОрТ 349876 4595") == "ПаСпОрТ <PASSPORT>"

    # С дополнительным текстом
    assert replace_passport("Мой паспорт: 349876 4595, выдан ОВД") == "Мой паспорт: <PASSPORT>, выдан ОВД"
    assert replace_passport(
        "Номер паспорта 349876 код 4595 действителен до 2025 года") == "Номер паспорта <PASSPORT> действителен до 2025 года"

    # Разные разделители
    assert replace_passport("паспорт №349876/4595") == "паспорт <PASSPORT>"
    assert replace_passport("паспорт №349876-4595") == "паспорт <PASSPORT>"

    # Границы слов
    assert replace_passport("Пример паспорта: 349876 4595.") == "Пример паспорта: <PASSPORT>."
    assert replace_passport("Серия и номер паспорта 349876 4595!") == "Серия и номер паспорта <PASSPORT>!"

    # Отсутствие данных
    assert replace_passport("У меня нет паспорта") == "У меня нет паспорта"
    assert replace_passport("Загранпаспорт 123456") == "Загранпаспорт 123456"

    # Крайние случаи
    assert replace_passport(
        "349876 4595 без слова паспорт") == "349876 4595 без слова паспорт"  # Нет ключевого слова
    assert replace_passport(
        "паспорт 123456 7890 с некорректными числами") == "паспорт <PASSPORT> с некорректными числами"

    # Длинный текст
    assert replace_passport(
        "Важная информация о паспорте: 349876 4595, который был выдан в 2020 году") == "Важная информация о паспорте: <PASSPORT>, который был выдан в 2020 году"

    # Базовые случаи
    print(replace_fioii("Директор Иванов И.И."))
    assert replace_fioii("Директор Иванов И.И.") == "<PER>"
    assert replace_fioii("Директор ИвановИ.И.") == "<PER>"
    print(replace_fioii("Директор: И.И.Иванов"))
    assert replace_fioii("Директор: И.И.Иванов") == "<PER>"

    # Разные разделители
    print(replace_fioii("Заместитель директора - Сидоров А.А."))
    assert replace_fioii("Заместитель директора - Сидоров А.А.") == "Заместитель директора - <PER>"
    print(replace_fioii("Главный инженер (Петров В.В.)"))
    assert replace_fioii("Главный инженер (Петров В.В.)") == "Главный инженер <PER>)"
    print(replace_fioii("Преподаватель, Кузнецов Д.М."))
    assert replace_fioii("Преподаватель, Кузнецов Д.М.") == "Преподаватель, <PER>"

    # С разными окончаниями фамилий
    assert replace_fioii("Начальник отдела Смирновой М.А.") == "Начальник отдела <PER>"
    assert replace_fioii("Старший менеджер Романова О.К.") == "Старший менеджер <PER>"

    # С двоеточием и запятыми
    assert replace_fioii("Ответственный: Попов С.С.") == "Ответственный: <PER>"
    print(replace_fioii("Автор книги: И.А. Николаев"))
    assert replace_fioii("Автор книги: И.А. Николаев") == "Автор книги: <PER>"

    # С заглавными буквами
    assert replace_fioii("ГЕНЕРАЛЬНЫЙ ДИРЕКТОР Путин В.В.") == "ГЕНЕРАЛЬНЫЙ ДИРЕКТОР <PER>"
    assert replace_fioii("ПОМОЩНИК: А.А. Михайлов") == "ПОМОЩНИК: <PER>"

    # С различными падежами
    assert replace_fioii("Договор с Ивановым И.И.") == "Договор с <PER>"
    assert replace_fioii("Письмо от Петрова А.А.") == "Письмо от <PER>"
    assert replace_fioii("Подпись под документом Сидоровым Д.Д.") == "Подпись под документом <PER>"

    # Сокращения должностей
    assert replace_fioii("Гл. бухгалтер Козлова Л.Л.") == "Гл. бухгалтер <PER>"
    print(replace_fioii("Техн. директор Васильев В.В."))
    assert replace_fioii("Техн. директор Васильев В.В.") == "Техн. <PER>"

    # С пробелами вокруг знаков препинания
    assert replace_fioii("Преподаватель , Иванов А.А.") == "Преподаватель , <PER>"
    assert replace_fioii("Ответственный : Петров В.В.") == "Ответственный : <PER>"

    # Фамилии с дефисом
    print(replace_fioii("Директор Петров-Васильев А.А."))
    assert replace_fioii("Директор Петров-Васильев А.А.") == "<PER> <PER>"
    assert replace_fioii("Главный специалист Иванова-Сидорова М.М.") == "Главный специалист <PER>"

    # Сложные предложения
    print(replace_fioii("Встреча с директором Ивановым И.И. прошла успешно."))
    assert replace_fioii(
        "Встреча с директором Ивановым И.И. прошла успешно.") == "Встреча с <PER>прошла успешно."
    print(replace_fioii(
        "Письмо от заместителя генерального директора Петрова А.А. получено."))
    assert replace_fioii(
        "Письмо от заместителя генерального директора Петрова А.А. получено.") == "Письмо от заместителя генерального <PER>получено."

    assert replace_org("ООО «Рога и копыта»") == "<ORG>"
    assert replace_org("ЗАО «Промышленная компания»") == "<ORG>"
    assert replace_org("ПАО «Газпром»") == "<ORG>"
    assert replace_org("ОАО «Роснефть»") == "<ORG>"
    assert replace_org("НКО «Благотворительный фонд»") == "<ORG>"
    assert replace_org("ИП Иванов И.И.") == "ИП Иванов И.И."
    assert replace_org("АО «Альфа-Банк»") == "<ORG>"
    assert replace_org("ГУП «Московский метрополитен»") == "<ORG>"
    assert replace_org("МУП «Водоканал»") == "<ORG>"
    assert replace_org("ФГУП «Почта России»") == "<ORG>"
    assert replace_org("общество с ограниченной ответственностью «Ромашка»") == "<ORG>"
    assert replace_org("закрытое акционерное общество «Стройка»") == "<ORG>"
    assert replace_org("публичное акционерное общество «Лукойл»") == "<ORG>"
    assert replace_org("открытое акционерное общество «Сбербанк»") == "<ORG>"
    assert replace_org("некоммерческая организация «Фонд поддержки»") == "<ORG>"
    assert replace_org("индивидуальный предприниматель Петров П.П.") == "индивидуальный предприниматель Петров П.П."
    assert replace_org("акционерное общество «Транснефть»") == "<ORG>"
    assert replace_org("государственное унитарное предприятие «Энергия»") == "<ORG>"
    assert replace_org("муниципальное унитарное предприятие «Тепло»") == "<ORG>"
    assert replace_org("федеральное государственное унитарное предприятие «Авиация»") == "<ORG>"

    assert replace_guid("Here is a GUID: 53c4cb70-edd0-4d55-9ad1-fffd207c2f23.") == "Here is a GUID: <GUID>."
    assert replace_guid(
        "First GUID: 53c4cb70-edd0-4d55-9ad1-fffd207c2f23, second GUID: 123e4567-e89b-12d3-a456-426614174000.") == "First GUID: <GUID>, second GUID: <GUID>."
    assert replace_guid("This text has no GUIDs.") == "This text has no GUIDs."
    assert replace_guid("53c4cb70-edd0-4d55-9ad1-fffd207c2f23") == "<GUID>"
    assert replace_guid("53c4cb70-edd0-4d55-9ad1-fffd207c2f23 123e4567-e89b-12d3-a456-426614174000") == "<GUID> <GUID>"
    assert replace_guid("53c4cb70-edd0-4d55-9ad1-fffd207c2f23 is at the start.") == "<GUID> is at the start."
    assert replace_guid(
        "The GUID is at the end: 53c4cb70-edd0-4d55-9ad1-fffd207c2f23") == "The GUID is at the end: <GUID>"
    assert replace_guid("This is not a GUID: 53c4cb70-edd0-4d55.") == "This is not a GUID: 53c4cb70-edd0-4d55."
    assert replace_guid(
        "GUID: 53c4cb70-edd0-4d55-9ad1-fffd207c2f23, another: 123e4567|e89b|12d3|a456|426614174000.") == "GUID: <GUID>, another: 123e4567|e89b|12d3|a456|426614174000."
    assert replace_guid("") == ""

    # Тест-кейс 1: Простой случай с "квартира"
    print(replace_apartment("Я живу в квартире 45."))
    assert replace_apartment("Я живу в квартире 45.") == "Я живу в кв. <ANUM>."

    # Тест-кейс 2: Сокращение "кв."
    assert replace_apartment(
        "Квартира находится на 12 этаже, это кв. 67.") == "Квартира находится на 12 этаже, это кв. <ANUM>."

    # Тест-кейс 3: Английское "apt."
    assert replace_apartment("My apartment is apt. 89.") == "My apartment is кв. <ANUM>."

    # Тест-кейс 4: Русское сокращение "к."
    assert replace_apartment("Моя квартира под номером к. 123.") == "Моя квартира под номером кв. <ANUM>."

    # Тест-кейс 5: Номер с буквами
    print(replace_apartment("Квартира 45A очень уютная."))
    assert replace_apartment("Квартира 45A очень уютная.") == "кв. <ANUM> очень уютная."

    # Тест-кейс 6: Английское "apartment"
    assert replace_apartment("I live in apartment 101.") == "I live in кв. <ANUM>."

    # Тест-кейс 7: Слово "комната" вместо "квартира"
    assert replace_apartment(
        "Эта комната 56 прекрасно подходит для работы.") == "Эта кв. <ANUM> прекрасно подходит для работы."

    # Тест-кейс 8: Разделительные пробелы вокруг номера
    assert replace_apartment("Квартира № 78 расположена на 5 этаже.") == "кв. <ANUM> расположена на 5 этаже."

    # Тест-кейс 9: Номер с символом "#"
    assert replace_apartment("Квартира 90 доступна для просмотра.") == "кв. <ANUM> доступна для просмотра."

    # Тест-кейс 10: Много слов перед номером
    assert replace_apartment("Моя личная квартира под номером 12B.") == "Моя личная квартира под номером 12B."

    # Тест-кейс 11: Английское "flat"
    assert replace_apartment("We are moving to flat 34 next month.") == "We are moving to кв. <ANUM> next month."

    # Тест-кейс 12: Несколько вариантов в одном тексте
    assert replace_apartment("Квартира 45 и кв. 67 обе хорошие.") == "кв. <ANUM> и кв. <ANUM> обе хорошие."

    # Тест-кейс 13: Номер с дефисом
    print(replace_apartment("Квартира 12-34 расположена в новом доме."))
    assert replace_apartment("Квартира 12-34 расположена в новом доме.") == "кв. <ANUM> расположена в новом доме."

    # Тест-кейс 14: Нижний регистр
    assert replace_apartment("квартира 78 находится рядом с парком.") == "кв. <ANUM> находится рядом с парком."

    # Тест-кейс 15: Номер в конце предложения
    assert replace_apartment("Выбранная квартира 99.") == "Выбранная кв. <ANUM>."

    # Тест-кейс 16: Номер внутри длинного слова
    assert replace_apartment("Квартира123") == "кв. <ANUM>"

    # Тест-кейс 17: Несколько вариантов написания
    assert replace_apartment("apt. 10, кв. 20, apartment 30.") == "кв. <ANUM>, кв. <ANUM>, кв. <ANUM>."

    # Тест-кейс 18: Номер с точкой
    assert replace_apartment("Квартира 45. Это отличный вариант.") == "кв. <ANUM>. Это отличный вариант."

    # Тест-кейс 19: Номер с запятой
    assert replace_apartment(
        "Квартира 67, расположенная в центре города, а старая квартира нет.") == "кв. <ANUM>, расположенная в центре города, а старая квартира нет."

    # Тест-кейс 20: Сложный текст с разными вариантами
    assert replace_apartment(
        "APT 89, КВ. 100, APPART 123, FLAT 45.") == "кв. <ANUM>, кв. <ANUM>, кв. <ANUM>, кв. <ANUM>."

    # Тест-кейс 1: Смешанные данные с телефонами, email и датами
    print(make_anonym("Контакты: +7(999)123-45-67, example@example.com, Дата рождения: 31.12.2000"))
    assert make_anonym("Контакты: +7(999)123-45-67, example@example.com, Дата рождения: 31.12.2000") == \
           "Контакты: <PHONE>, <EMAIL>, Дата рождения: <DATE>"

    # Тест-кейс 2: Финансовые данные с номерами и датами
    print(make_anonym("Счет №123456, открытие 2023-01-15, баланс 10000 рублей"))
    assert make_anonym("Счет №123456, открытие 2023-01-15, баланс 10000 рублей") == \
           "Счет №123456, открытие <DATE>, баланс <SUM> рублей"

    # Тест-кейс 3: Разнообразные форматы телефонов и <SNILS>
    print(make_anonym("Телефоны: 8-999-123-45-67, +7 (987) 654-32-10, СНИЛС: 123-456-789 01"))
    assert make_anonym("Телефоны: 8-999-123-45-67, +7 (987) 654-32-10, СНИЛС: 123-456-789 01") == \
           "Телефоны: <PHONE>, <PHONE>, СНИЛС: <SNILS>"

    # Тест-кейс 4: Email, телефоны и денежные суммы
    assert make_anonym(
        "Заказ №777, оплачен на сумму 5000 рублей, контакт: user.name+tag@domain.com, тел.: 89991234567") == \
           "Заказ №777, оплачен на сумму <SUM> рублей, контакт: <EMAIL>, тел.: <PHONE>"

    # Тест-кейс 5: Сложный текст с различными форматами данных
    print(make_anonym("Отчет за период 01.01.2023 - 31.12.2023, сумма: 12345.67 €, ответственный: +7(999)999-99-99",
                      'token'))
    assert make_anonym("Отчет за период 01.01.2023 - 31.12.2023, сумма: 12345.67 €, ответственный: +7(999)999-99-99") == \
           "Отчет за период <DATE> - <DATE>, сумма: <SUM> €, ответственный: <PHONE>"

    # Тест-кейс 6: Комбинация всех типов данных
    print(make_anonym(
        "Договор №12345 от 2023-05-15, клиент: Ivanov I.I., тел.: +7(999)123-45-67, email: ivanov@mail.ru, СНИЛС: 123-456-789 01, сумма: 100000 руб."))
    assert make_anonym(
        "Договор №12345 от 2023-05-15, клиент: Ivanov I.I., тел.: +7(999)123-45-67, email: ivanov@mail.ru, СНИЛС: 123-456-789 01, сумма: 100000 руб.") == \
           "Договор №12345 от <DATE>, клиент: Ivanov I.I., тел.: <PHONE>, email: <EMAIL>, СНИЛС: <SNILS>, сумма: <SUM> руб."

    # Тест-кейс 7: Несколько дат разных форматов и телефонов
    print(make_anonym(
        "Встреча назначена на 15/05/2023 в 14:00, напомнить за день: 2023-05-14, контакты: +79991234567, (8-987)654-32-10"))
    assert make_anonym(
        "Встреча назначена на 15/05/2023 в 14:00, напомнить за день: 2023-05-14, контакты: +79991234567, (8-987)654-32-10") == \
           "Встреча назначена на <DATE> в 14:00, напомнить за день: <DATE>, контакты: <PHONE>, ( <PHONE>"

    # Тест-кейс 8: Много различных числовых значений
    print(make_anonym("Код заказа: 123456, количество: 7 шт., стоимость: 12345.67 руб., срок доставки: 31.12.2023"))
    assert make_anonym("Код заказа: 123456, количество: 7 шт., стоимость: 12345.67 руб., срок доставки: 31.12.2023") == \
           "Код заказа: 123456, количество: 7 шт., стоимость: <SUM> руб., срок доставки: <DATE>"

    # Тест-кейс 9: Сложная структура с вложенными данными
    print(make_anonym(
        "Филиал №123, адрес: г. Москва, ул. Примерная, д. 12, тел.: +7(495)123-45-67, email: office@company.ru, директор: Иванов И.И., СНИЛС: 987-654-321 01"))
    assert make_anonym(
        "Филиал №123, адрес: г. Москва, ул. Примерная, д. 12, комн 9 тел.: +7(495)123-45-67, email: office@company.ru, директор: Иванов И.И., СНИЛС: 987-654-321 01") == \
           "<ORG><0>, адрес: г. <LOC><0>, ул. <STREET>, д. <BNUM>, кв. <ANUM> тел.: <PHONE>, email: <EMAIL>, директор: <PER><0>, СНИЛС: <SNILS>"

    # Тест-кейс 10: Экспортные данные с множеством параметров
    print(make_anonym(
        "Экспортная накладная №789456 от 2023-07-15, получатель: John Doe, тел.: +1 (202) 555-0133, email: john.doe@example.com, сумма: 12345.67 $, СНИЛС отправителя: 654-321-098 01"))
    assert make_anonym(
        "Экспортная накладная №123456 от 2023-07-15, получатель: John Doe, тел.: +1 (202) 555-0133, email: john.doe@example.com, сумма: 12345.67 $, СНИЛС отправителя: 654-321-098 01") == \
           "Экспортная накладная №123456 от <DATE>, получатель: John Doe, тел.: <PHONE>, email: <EMAIL>, сумма: <SUM> $, СНИЛС отправителя: <SNILS>"

    print(make_anonym(
        "Экспортная накладная №789456 от 2023-07-15, получатель: John Doe, тел.: +1 (202) 555-0133, email: john.doe@example.com, сумма: 12345.67 $, СНИЛС отправителя: 654-321-098 01",
        'fake'))

def _test_all():
    test_all()


def main():
    msg = 'usage: python anonym.py <input file_name> <mode: token/fake - default token (optional)>'
    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            _test_all()
            exit(0)
        else:
            if len(sys.argv) > 3:
                print('too many arguments. ' + msg)

            file_name = sys.argv[1]
            if not os.path.isfile(file_name):
                print(f'file {file_name} not found')
                exit(1)
            else:
                with open(file_name, 'r') as fd:
                    data = fd.read()
            if len(sys.argv) == 3:
                mode = sys.argv[2]
                if mode != 'token' and mode != 'fake':
                    print('available modes: [token, fake]. unknown value for mode: ' + mode)
                    exit(1)
            else:
                mode = 'token'
            anonymized = make_anonym(data, mode)
            print(anonymized)
            exit(0)
    else:
        print(msg)


if __name__ == "__main__":
    main()
