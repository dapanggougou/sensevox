**SenseVox**는 [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), [FunASR SenseVoice](https://github.com/FunAudioLLM/SenseVoice), 그리고 [flet](https://github.com/flet-dev/flet/)을 활용한 Windows 기반 오프라인 음성 인식 도구입니다. 이 프로젝트는 사용자에게 효율적이고 편리한 오프라인 음성 인식 경험을 제공하는 것을 목표로 합니다.

<img src="https://github.com/user-attachments/assets/84f46047-d144-4cc3-976b-24670f66e463" alt="예시 이미지" width="250"/>

## 주요 기능

- **오프라인 음성 인식:** 데이터 프라이버시 보호
- **다국어 지원:** 중국어(표준어), 광둥어, 영어, 일본어, 한국어
- **내장 200MB 모델:** 더 큰 모델로 교체하여 인식 정확도 향상 가능
- **간편한 사용:** 다운로드, 압축 해제, 실행 3단계
- **기술 기반:** sherpa-onnx와 FunASR SenseVoice 기술 채택
- **호환성:** Windows 10 지원(다른 버전은 미검증)

## 사용 방법

1. **압축 파일 다운로드**
   프로젝트의 [Releases](https://github.com/dapanggougou/sensevox/releases) 페이지에서 최신 버전 압축 파일 다운로드.

2. **압축 해제**
   다운로드한 압축 파일을 원하는 디렉토리에 압축 해제 **(주의: 경로는 모두 영문이어야 함)**.

3. **애플리케이션 실행**
   **sensevox.exe** 실행.

4. **핫키 설정**
   - Set Hotkey를 마우스 클릭 후 원하는 키 입력.
   - 기본 핫키는 **스페이스 바 (Space)**
   - **Caps Lock 키**로 변경 권장
   - 핫키는 조합 키도 가능.

5. **시스템 트레이/백그라운드 실행**
   - [Traymond](https://github.com/fcFn/traymond) 또는 [RBTray](https://sourceforge.net/projects/rbtray/) 등 서드파티 소프트웨어로 시스템 트레이 최소화 가능.
   - **Win+Tab** 또는 **터치패드 멀티핑거 위로 스와이프**로 별도 데스크톱 이동 가능.

## 녹음 장치 변경 방법

Windows 설정에서 다음 절차로 장치 선택 가능:
1. **Windows 설정** 열기.
2. **시스템** > **소음** 선택.
   또는 스피커 아이콘 우클릭 후 **소리 설정 열기**로 빠르게 접근 가능.

미검증 상태이므로 효과 불확실.

## 모델 파일 설명

- 압축 파일에는 기본적으로 200MB 모델 포함.
- 인식 정확도 향상을 위해 900MB 모델이 필요한 경우, 아래에서 다운로드:
  - **[모델 링크](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2)**
- 다운로드 후 `model.onnx` 파일 교체:
  - 저장 경로: `_internal/assets/sensevoicesmallonnx/model.onnx`

---

### 기타 사항
여러 모델 지원 예정인가요? 없습니다.

whisper의 경우 whisper.cpp의 vulkan 버전을 테스트했으나, smallq5 모델에서도 0.6초의 지연이 발생해 실시간 입력이 불가능했습니다. GPU 가속 버전은 범용성이 떨어지고 패키지 크기도 커져 채택하지 않았습니다.

paraformer-zh는 중국어에서 약간 우수하지만, 악센트와 마이크 수음 상태가 양호하면 효과는 비슷하며 모두 높은 정확도를 보입니다. 따라서 추가 계획 없음.

sensevoice.cpp 사용도 고려했지만, Python 구현 방법을 알 수 없었습니다.

---

다시 한번, FunASR SenseVoice, sherpa-onnx, CapsWriter-Offline, flet, Gemini 2.5 Pro에 감사드립니다.
