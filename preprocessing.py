import googletrans as gt
import random
from koeda import EasyDataAugmentation, AEasierDataAugmentation


def Translator_Augmentation(text, language=None, repetition=2):
    """
    input :
        - text
        - language : 언어 선택. 선택하지 않으면 랜덤
        - repetition : default = 2

    주어진 텍스트를 선택한 언어로 번역 후 한국어로 재번역하는 함수입니다.

    output :
        - List
    """
    translator = gt.Translator()
    result = []
    keys = list(gt.LANGUAGES.keys())

    for _ in range(repetition):
        if language is None:
            select_language = random.choice(keys)
        else:
            select_language = language
        translated = translator.translate(text, src="ko", dest=select_language)
        re_translated = translator.translate(
            translated.text, src=select_language, dest="ko"
        )
        result.append(re_translated.text)

    return result


ANALYZER = ["Okt", "Kkma", "Komoran", "Hannanum"]


def AEDA_Augmentation(
    text,
    morpheme_analyzer=None,
    repetition=1,
    punctuations=[".", ",", "!", "?", ";", ":"],
):
    """
    input :
        - text
        - morpheme_analyzer : 형태소 분석기 선택. 선택하지 않으면 랜덤
        - repetition : default = 1
        - punctuation : default = [".", ",", "!", "?", ";", ":"]

    주어진 텍스트를 선택한 형태소 분석기로 분리한 후 그 사이에 랜덤하게 주어진 문장부호를 추가하는 함수입니다.

    output :
        - List or str
    """
    if morpheme_analyzer is None:
        morpheme_analyzer = random.choice(ANALYZER)
    aeda = AEasierDataAugmentation(
        morpheme_analyzer=morpheme_analyzer, punctuations=punctuations
    )
    result = aeda(text, repetition=repetition)
    return result


class Custom_EasyDataAugmentation(EasyDataAugmentation):
    def _eda(self, data, p):
        select = 0
        for i in range(4):
            if self.alphas[i] == 1:
                select = i
        augmented_sentences = self.augmentations[select](data, p[select])

        return augmented_sentences


def EDA_Augmentation(
    text, morpheme_analyzer=None, select="ri", p=(0.3, 0.3, 0.3, 0.3), repetition=1
):
    """
    input :
        - text
        - morpheme_analyzer : 형태소 분석기 선택. 선택하지 않으면 랜덤
        - select : 어떤 방식을 사용할지 선택. default = 'ri'
        - p : 비율. default = (0.3, 0.3, 0.3, 0.3)
        - repetition : default = 1

    주어진 텍스트를 선택한 형태소 분석기로 분리한 후 선택한 방식을 적용하는 함수입니다.
    <선택할 수 있는 방식>
    - sr : 유의어로 교체
    - ri : 랜덤 단어 삽입
    - rs : 랜덤한 두 단어의 위치 교환
    - rd : 랜덤 단어 삭제

    output :
        - List or str
    """
    if morpheme_analyzer is None:
        morpheme_analyzer = random.choice(ANALYZER)
    if select == "sr":
        a = (1, 0, 0, 0)
    elif select == "ri":
        a = (0, 1, 0, 0)
    elif select == "rs":
        a = (0, 0, 1, 0)
    else:  # rd
        a = (0, 0, 0, 1)

    eda = Custom_EasyDataAugmentation(
        morpheme_analyzer=morpheme_analyzer,
        alpha_sr=a[0],
        alpha_ri=a[1],
        alpha_rs=a[2],
        prob_rd=a[3],
    )

    result = eda(text, p=p, repetition=repetition)
    return result
