<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-FF6F00?style=flat-square)](https://mediapipe.dev/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00BFFF?style=flat-square)](https://ultralytics.com/)

<h4>VisionHub: Um hub interativo de visão computacional com IA em tempo real</h4>

<a href="#sobre-o-projeto">Sobre</a> •
<a href="#objetivo">Objetivo</a> •
<a href="#como-funciona">Como funciona</a> •
<a href="#módulos">Módulos</a> •
<a href="#instalação">Instalação</a> •
<a href="#uso">Uso</a> •
<a href="#estrutura">Estrutura</a> •
<a href="#dependências">Dependências</a>

</div>

---

## Sobre o projeto

VisionHub é um sistema interativo de visão computacional que reúne múltiplos módulos de IA operando sobre a webcam em tempo real. A ideia central é funcionar como um "hub" de inteligência artificial: cada módulo demonstra, de forma prática e visual, como algoritmos interpretam o comportamento humano e o ambiente ao redor.

O sistema foi desenvolvido como uma plataforma de demonstração, uma ferramenta para explorar e entender como técnicas de visão computacional funcionam na prática.

## Objetivo

> **Promover o letramento em Inteligência Artificial.**

Permitir que qualquer pessoa veja, em tempo real e sem precisar entender o código, como algoritmos de IA analisam um rosto, classificam uma emoção ou identificam objetos num ambiente. O VisionHub torna conceitos abstratos de IA tangíveis e observáveis.

## Como funciona

Uma janela pygame exibe o feed da câmera com uma sidebar lateral. Cada módulo é ativado ou desativado com uma tecla e desenha seus resultados diretamente no frame, sem janelas separadas.

## Módulos

### Foco (`1`) — Sonolência e atenção
Detecta sinais de cansaço e distração usando landmarks faciais do MediaPipe.

- **Sonolência:** calcula o EAR (Eye Aspect Ratio) — se os olhos ficarem fechados por `N` frames consecutivos, exibe alerta
- **Distração:** estima a pose da cabeça (yaw e pitch) via solvePnP — alerta se o rosto sair dos ângulos configurados
- Exibe as métricas de EAR, yaw e pitch em cards no canto do frame

### Emoção (`2`) — Reconhecimento facial
Classifica a emoção predominante do rosto usando um modelo ONNX leve (FERPlus).

- Detecta o rosto com Haar Cascade, recorta o ROI e roda a inferência
- Exibe a emoção dominante com confiança e barras para cada classe
- Processa a cada `N` frames para manter o loop fluido (configurável)

**Classes:** neutro · feliz · surpreso · triste · raiva

### Objetos (`3`) — Detecção no ambiente
Detecta objetos relevantes no frame com YOLOv8n.

- Filtra apenas itens de uma lista predefinida: celular, laptop, livro, copo, mochila etc.
- Aplica threshold de confiança configurável
- Exibe caixas HUD com label e confiança de cada objeto detectado

## Instalação

```bash
git clone https://github.com/ianderichalski/vision-hub.git
cd vision-hub

python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

> O modelo ONNX de emoção e o peso `yolov8n.pt` são baixados automaticamente na primeira execução de cada módulo.

## Uso

```bash
python app.py
```

| Tecla | Ação |
|---|---|
| `1` | Liga/desliga Foco |
| `2` | Liga/desliga Emoção |
| `3` | Liga/desliga Objetos |
| `Q` | Sair |

Para ajustar câmera, resolução ou thresholds, edite `config.py`.

## Estrutura

```
vision-hub/
├── app.py              # loop principal, janela pygame e sidebar
├── config.py           # todos os parâmetros (câmera, thresholds, cores)
├── requirements.txt
└── modules/
    ├── ui.py           # helpers de desenho: pill, bar, hud_box, alert_banner
    ├── focus.py        # EAR + pose da cabeça via MediaPipe
    ├── emotion.py      # classificação com modelo ONNX FERPlus
    └── objects.py      # detecção YOLOv8n
```

## Dependências

| Biblioteca | Finalidade |
|---|---|
| `opencv-python` | Captura de câmera, desenho e pré-processamento |
| `mediapipe` | Landmarks faciais para detecção de foco |
| `ultralytics` | Modelo YOLOv8n para detecção de objetos |
| `scipy` | Cálculo de distância euclidiana (EAR) |
| `numpy` | Operações matriciais |
| `pygame` | Janela principal e sidebar interativa |