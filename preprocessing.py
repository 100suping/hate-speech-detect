import googletrans as gt
import random
from koeda import EasyDataAugmentation, AEasierDataAugmentation

def Translator_Augmentation(text, language=None, repetition=2):
    '''
        input : 
            - text 
            - language : 언어 선택. 선택하지 않으면 랜덤
            - repetition : default = None 
    
        output :
            - List
    '''
    translator = gt.Translator()
    result = []
    keys = list(gt.LANGUAGES.keys())

    for _ in range(repetition):
        if language is None:
            select_language = random.choice(keys)
        else:
            select_language = language
        translated = translator.translate(text, src='ko', dest=select_language)
        re_translated = translator.translate(translated.text, src=select_language, dest='ko')
        result.append(re_translated.text)
    
    return result

ANALYZER = ["Okt", "Kkma", "Komoran", "Hannanum"]

def AEDA_Augmentation(text, morpheme_analyzer=None, repetition=1, punctuations=[".", ",", "!", "?", ";", ":"]):
    if morpheme_analyzer is None:
        morpheme_analyzer = random.choice(ANALYZER)
    aeda = AEasierDataAugmentation(
        morpheme_analyzer=morpheme_analyzer, punctuations=punctuations
    )
    result = aeda(text, repetition=repetition)
    return(result)