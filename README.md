"# nlp" 

"## 형태소 분석기 MeCab 설치"

"1. mecab-ko-msvc 설치하기

1-1. 링크[https://github.com/Pusnow/mecab-ko-msvc/releases/tag/release-0.9.2-msvc-3]에 들어가 본인의 윈도우 버전에 마줘 32bit/64bit 선택해 다운로드한다.

1-2. C드라이브에 mecab 폴더를 만들고 위에서 다운로드 받은 압축파일을 푼다.

 

2. mecab-ko-dic-msvc.zip 기본 사전 설치하기

2-1. 마찬가지로, 링크[https://github.com/Pusnow/mecab-ko-dic-msvc/releases/tag/mecab-ko-dic-2.1.1-20180720-msvc]에 들어가 사전을 다운받는다.

2-2. 1-2에서 만든 C드라이브 mecab 폴더에 들어가 다운 받은 압축파일을 푼다.

3. python wheel 설치

3-1. 링크[https://github.com/Pusnow/mecab-python-msvc/releases/tag/mecab_python-0.996_ko_0.9.2_msvc-2]에 접속해 자신의 파이썬 버전과 윈도우 버전에 맞는 whl을 다운로드 한다.

3-2. 다운로드 파일을 mecab을 실행시킬 가상환경에 들어가

pip install mecab_python-0.996_ko_0.9.2_msvc-cp38-cp38-win_amd64.whl

"


