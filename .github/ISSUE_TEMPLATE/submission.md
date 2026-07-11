---
name: Сдача тестового задания
about: Форма для отправки выполненного MTBank AI Engineer test assignment
title: "[SUBMISSION] Имя Фамилия — MTBank AI Transcription"
labels: submission
assignees: ''
---

## Кандидат

**Имя Фамилия: Кирилл Булавин**  
**Email: bulavinkirill02@gmail.com**  
**Telegram / телефон (опционально): @bvlkr / 80336690129**  

---

## Ссылки

**GitHub репозиторий: https://github.com/JustiZzZz/mtbank-ai-transcription**  
**Живое демо (HTTPS): https://165.227.237.72.sslip.io/**  
**OpenWebUI model name:** `mtbank_ai_transcription`  


```text
Проанализируй загруженный звонок
```

---

## Архитектура

```text
Решение построено вокруг общего AnalysisService, который используется и FastAPI
/analyze, и OpenWebUI Pipeline. Аудио нормализуется через ffmpeg,
транскрибируется faster-whisper medium, размечается fallback diarizer, после
чего четыре LLM-агента на OpenAI-compatible API выполняют классификацию,
оценку качества, compliance-анализ и суммаризацию. Все ответы валидируются
Pydantic-моделями, каждый агент имеет deterministic fallback, а production demo
развернуто через Docker Compose и HTTPS reverse proxy.
```

---

## Реализованные компоненты

- [X] OpenWebUI Pipeline
- [X] REST API `POST /analyze`
- [X] ASR на `faster-whisper` / `openai-whisper`
- [X] Поддержка минимум WAV/MP3/OGG
- [X] Базовая диаризация `Оператор` / `Клиент`
- [X] Classifier agent
- [X] Quality agent
- [X] Compliance agent
- [X] Summarizer agent
- [X] Docker Compose
- [X] `.env.example`
- [X] JSON-логи входа/выхода каждого агента
- [X] Unit tests для агентов
- [X] Integration test Pipeline/API
- [X] WER/CER таблица
- [X] HTTPS demo

---

## Результаты ASR

Смотреть в README.md

```text
Пример:
Mean WER: 0.036
Mean CER: 0.013
ASR: faster-whisper medium
Device: CPU int8
```

---
## Тестовые данные

- [X] 6 аудиофайлов русской речи
- [X] Есть эталонные транскрипты
- [X] Есть минимум один 8 kHz / phone-quality файл
- [X] Есть диалог двух говорящих длительностью 1+ минута
- [X] Общая длительность 5+ минут
---

## Затраченное время

- [X] 16–24 часа (2–3 дня)


