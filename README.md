# White Vision

White Vision은 시각 장애인과 저시력자, 노인의 안전하고 독립적인 보행을 지원하는 AI 기반 시각 보조 서비스입니다. 이 프로젝트는 MetaLense2 XR Glass를 활용하여 저시력자와 노인에게 보행 중 발생할 수 있는 다양한 위험 요소를 사전에 인지하고 회피할 수 있도록 지원하는 시스템을 개발하였습니다.

## 프로젝트 개요
White Vision은 AI 기반 이미지 세그멘테이션 기술을 사용하여 보행로와 위험 요소를 인식하고, AR(증강 현실) 기술을 통해 사용자에게 실시간으로 안전 정보를 제공합니다. 이를 통해 시각 장애인 및 저시력자들이 보행 시 직면하는 문제들을 해결하고, 안전하고 자율적인 보행을 가능하게 합니다.

## 주요 기능
1. **지면 구분 색상 표시**: 인도, 도로, 자전거 도로 등 다양한 지면을 명확히 구분하여 색상으로 표시합니다. 안전한 구역(인도, 점자 블록)과 위험 구역(도로, 자전거 도로)을 대비되는 색상으로 구분하여 안전한 보행을 돕습니다.
2. **장애물 인식 및 경고**: 킥보드, 자전거, 사람 등 충돌 위험이 있는 장애물을 실시간으로 인식하고, 색상 대비와 음성 알림으로 사용자에게 경고를 제공합니다.
3. **경계턱 및 도로 파임 경고**: ARCore SDK의 Depth API를 활용하여 경계턱, 파임 구멍 등 높이 차이가 있는 위험 요소를 실시간으로 감지하고 경고합니다.

## 기술 설명
- **이미지 세그멘테이션 모델**: Computer Vision 기술을 활용해 다양한 도로 및 지면 종류를 실시간으로 구분합니다.
- **Object Detection**: 장애물 요소를 감지하고, AR을 통해 시각적 경고와 음성 피드백을 제공합니다.
- **증강 현실**: MetaLense2의 AR 기술을 활용해 지면 정보를 시각적으로 덧대어 사용자가 직관적으로 위험을 인식할 수 있도록 합니다.

## 기대 효과 및 활용 방안
White Vision은 시각 장애인과 저시력자, 노인의 안전한 보행을 지원하고, 도로 환경 개선을 위한 데이터를 제공할 수 있습니다. 또한, 위험 요소에 대한 시각적 정보를 통해 교육 자료로도 활용될 수 있으며, 보행 안전에 대한 인식을 높이는 데 기여할 수 있습니다.

## 팀 정보
- **팀명**: 주동자들
- **팀원**:
  - 장소현 (팀장, 기획)
  - 임채윤 (디자인, XR)
  - 서동주 (AI)

## 사용 기술
- **장비**: MetaLense2 XR Glass
- **데이터셋**: AI Hub에서 제공하는 '도보 보행 영상' 데이터셋
- **프레임워크**: ARCore SDK, MetaCore 2.0 SDK, Computer Vision 기술

## 설치 및 실행 방법
1. 이 저장소를 클론합니다:
   ```sh
   git clone https://github.com/seodj01/white_vision_capstone.git
   ```
2. MetaLense2와 함께 Android 환경에서 ARCore SDK를 설치하고, 프로젝트 코드를 빌드하여 실행합니다.

## 라이선스
본 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 LICENSE 파일을 참고하세요.