"""Generate Russian MTBank-style phone-call test audio.

Requires:
    py -m pip install edge-tts
    ffmpeg and ffprobe available in PATH
"""

# ruff: noqa: E501

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import edge_tts

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "test_data"
BUILD_DIR = OUT_DIR / "_build"
OPERATOR_VOICE = "ru-RU-SvetlanaNeural"
CLIENT_VOICE = "ru-RU-DmitryNeural"
PAUSE_SECONDS = 0.65
PHONE_FILTER = "highpass=f=300,lowpass=f=3400,acompressor=threshold=-18dB:ratio=2.5:attack=8:release=120"


Speaker = Literal["Оператор", "Клиент", "UNKNOWN"]


@dataclass(frozen=True)
class Turn:
    speaker: Speaker
    text: str


@dataclass(frozen=True)
class Scenario:
    slug: str
    title: str
    output_format: str
    codec_args: tuple[str, ...]
    expected_topic: str
    expected_priority: str
    expected_compliance_passed: bool
    notes: str
    turns: tuple[Turn, ...]


SCENARIOS = (
    Scenario(
        slug="call01_credit_online",
        title="Кредит онлайн: нормальный качественный звонок",
        output_format="wav",
        codec_args=("-ar", "8000", "-ac", "1", "-c:a", "pcm_mulaw"),
        expected_topic="кредиты",
        expected_priority="medium",
        expected_compliance_passed=True,
        notes="Хороший операторский чеклист, понятные action items.",
        turns=(
            Turn("Оператор", "Добрый день, МТБанк, меня зовут Анна, чем могу помочь?"),
            Turn("Клиент", "Здравствуйте. Хочу узнать условия кредита наличными."),
            Turn("Оператор", "Конечно. Подскажите, какая сумма и на какой срок вас интересует?"),
            Turn("Клиент", "Около двенадцати тысяч рублей на два года. Я получаю зарплату на карту МТБанка."),
            Turn("Оператор", "Поняла. Для зарплатных клиентов заявка рассматривается индивидуально, решение принимается после проверки анкеты и документов."),
            Turn("Клиент", "Мне важно понимать ежемесячный платеж и можно ли погасить досрочно."),
            Turn("Оператор", "Предварительно можно рассчитать платеж в приложении. Досрочное погашение доступно без штрафа, но итоговые условия будут в договоре."),
            Turn("Клиент", "Хорошо, я попробую оформить через приложение, но боюсь ошибиться в анкете."),
            Turn("Оператор", "Я могу отправить вам инструкцию по заполнению заявки и список документов на email."),
            Turn("Клиент", "Да, отправьте, пожалуйста."),
            Turn("Оператор", "Хорошо, после звонка отправлю инструкцию. Есть ли еще вопросы по кредиту?"),
            Turn("Клиент", "Нет, спасибо, все понятно."),
            Turn("Оператор", "Спасибо за обращение в МТБанк, хорошего дня."),
        ),
    ),
    Scenario(
        slug="call02_halva_fraud",
        title="Халва: блокировка, подозрение на мошенничество",
        output_format="mp3",
        codec_args=("-ar", "8000", "-ac", "1", "-b:a", "32k"),
        expected_topic="жалобы",
        expected_priority="high",
        expected_compliance_passed=True,
        notes="Граничный случай: карта/Халва плюс fraud-risk и срочная блокировка.",
        turns=(
            Turn("Оператор", "Здравствуйте, МТБанк, оператор Ирина. Чем могу помочь?"),
            Turn("Клиент", "У меня карта Халва, только что пришло списание, которое я не совершал."),
            Turn("Оператор", "Понимаю. Сейчас важно заблокировать карту. Подтвердите, пожалуйста, вы видите операцию в приложении?"),
            Turn("Клиент", "Да, сумма небольшая, но я боюсь, что дальше спишут больше."),
            Turn("Оператор", "Я не буду просить у вас ПИН, CVC или коды из SMS. Эти данные никому сообщать нельзя."),
            Turn("Клиент", "Хорошо. Что делать дальше?"),
            Turn("Оператор", "Заблокируйте карту в приложении или я подскажу путь. Затем нужно оформить обращение по спорной операции."),
            Turn("Клиент", "Я уже нажал блокировку, но хочу оставить жалобу, потому что поддержка долго отвечала."),
            Turn("Оператор", "Жалобу зарегистрируем отдельно. По спорной операции специалист проверит детали и сроки рассмотрения."),
            Turn("Клиент", "Мне нужен новый пластик, потому что я завтра уезжаю."),
            Turn("Оператор", "Я передам запрос на срочный перевыпуск и отправлю инструкцию по цифровой карте."),
            Turn("Клиент", "Спасибо. Главное, чтобы никто больше не списал деньги."),
            Turn("Оператор", "После блокировки операции по карте невозможны. Спасибо за обращение, хорошего дня."),
        ),
    ),
    Scenario(
        slug="call03_transfer_stuck",
        title="Перевод: деньги не дошли, реквизиты и ЕРИП",
        output_format="ogg",
        codec_args=("-ar", "8000", "-ac", "1", "-c:a", "libvorbis", "-q:a", "2"),
        expected_topic="переводы",
        expected_priority="medium",
        expected_compliance_passed=True,
        notes="Проверяет переводы, платежи, реквизиты и задержку зачисления.",
        turns=(
            Turn("Оператор", "Добрый день, МТБанк, меня зовут Анна. Какой у вас вопрос?"),
            Turn("Клиент", "Вчера сделал перевод по реквизитам, деньги списались, но получатель их не видит."),
            Turn("Оператор", "Уточните, пожалуйста, это был платеж через ЕРИП, перевод на карту или банковский перевод по IBAN?"),
            Turn("Клиент", "Это был банковский перевод по IBAN на счет другой организации."),
            Turn("Оператор", "Поняла. Такие платежи могут идти дольше, особенно если реквизиты требуют дополнительной проверки."),
            Turn("Клиент", "В назначении платежа я написал номер договора. Может быть ошибка в реквизитах?"),
            Turn("Оператор", "Мы можем проверить статус платежа по квитанции. Если реквизиты неверные, возможен возврат от банка получателя."),
            Turn("Клиент", "Комиссия тоже списалась. Ее вернут?"),
            Turn("Оператор", "Возврат комиссии зависит от причины отклонения. Я создам обращение на проверку платежа."),
            Turn("Клиент", "Мне нужен ответ сегодня, потому что организация ждет оплату."),
            Turn("Оператор", "Я отмечу срочность и передам обращение специалистам. Номер обращения отправим в SMS."),
            Turn("Клиент", "Хорошо, жду SMS."),
            Turn("Оператор", "Спасибо за обращение. Если платеж зачислится раньше, вы увидите статус в интернет-банке."),
        ),
    ),
    Scenario(
        slug="call04_compliance_risky",
        title="Кредит: рискованные обещания оператора",
        output_format="wav",
        codec_args=("-ar", "8000", "-ac", "1", "-c:a", "pcm_s16le"),
        expected_topic="кредиты",
        expected_priority="medium",
        expected_compliance_passed=False,
        notes="Должен сработать compliance: гарантии одобрения и превосходные степени.",
        turns=(
            Turn("Оператор", "Добрый день, это МТБанк, специалист Ольга."),
            Turn("Клиент", "Мне нужен кредит, но у меня уже есть задолженность в другом банке."),
            Turn("Оператор", "Не переживайте, мы гарантируем одобрение почти каждому клиенту."),
            Turn("Клиент", "Точно? Мне уже один банк отказал."),
            Turn("Оператор", "У нас лучшие условия на рынке и ставка самая выгодная, поэтому заявку точно одобрят."),
            Turn("Клиент", "А документы нужны?"),
            Turn("Оператор", "Можно начать без проверки, потом все донесете. Главное подать заявку прямо сейчас."),
            Turn("Клиент", "Мне нужно подумать. Я не хочу лишних комиссий."),
            Turn("Оператор", "Скрытых комиссий нет, без переплат, просто оформим и все."),
            Turn("Клиент", "Отправьте тогда условия, я сравню."),
            Turn("Оператор", "Хорошо, отправлю условия и ссылку на заявку после звонка."),
        ),
    ),
    Scenario(
        slug="call05_poor_quality_complaint",
        title="Плохое качество консультации: жалоба клиента",
        output_format="mp3",
        codec_args=("-ar", "8000", "-ac", "1", "-b:a", "24k"),
        expected_topic="жалобы",
        expected_priority="high",
        expected_compliance_passed=True,
        notes="Плохой чеклист: нет нормального приветствия, слабое выявление потребности.",
        turns=(
            Turn("Оператор", "Да, слушаю."),
            Turn("Клиент", "Здравствуйте. Я уже третий раз звоню по поводу блокировки карты."),
            Turn("Оператор", "Что у вас там?"),
            Turn("Клиент", "Карту заблокировали после поездки, приложение пишет обратиться в банк."),
            Turn("Оператор", "Ну значит ждите, система сама проверит."),
            Turn("Клиент", "Но мне надо оплатить гостиницу сегодня. Можно ускорить?"),
            Turn("Оператор", "Я не знаю, попробуйте позже."),
            Turn("Клиент", "Это жалоба. Мне не объяснили причину и не предложили решение."),
            Turn("Оператор", "Можете написать в чат."),
            Turn("Клиент", "Запишите обращение и передайте supervisor, пожалуйста."),
            Turn("Оператор", "Ладно, передам обращение специалисту."),
            Turn("Клиент", "Номер обращения будет?"),
            Turn("Оператор", "Придет SMS."),
        ),
    ),
    Scenario(
        slug="call06_unknown_nonsense",
        title="Бессмысленный разговор: должен остаться UNKNOWN/другое",
        output_format="flac",
        codec_args=("-ar", "8000", "-ac", "1", "-c:a", "flac"),
        expected_topic="другое",
        expected_priority="low",
        expected_compliance_passed=True,
        notes="Нет банковской темы и нет операторского opener; fallback diarizer должен оставить UNKNOWN.",
        turns=(
            Turn("UNKNOWN", "Алло. У меня на столе зеленая лампа и три пустые коробки."),
            Turn("UNKNOWN", "А у меня чайник спорит с календарем, потому что сегодня вторник или облако."),
            Turn("UNKNOWN", "Вы слышите, как карандаш считает ступеньки?"),
            Turn("UNKNOWN", "Слышу, но только если окно смотрит на северный суп."),
            Turn("UNKNOWN", "Тогда положите круглый билет рядом с тишиной."),
            Turn("UNKNOWN", "Я положил, но он стал прозрачным и начал шептать про апельсины."),
            Turn("UNKNOWN", "Хорошо. На этом странный разговор можно закончить."),
            Turn("UNKNOWN", "Да, до свидания, квадратная погода."),
        ),
    ),
)


def run(command: list[str]) -> None:
    subprocess.run(command, check=True, capture_output=True, text=True)


def ffprobe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


async def synthesize_turn(turn: Turn, output_path: Path) -> None:
    voice = OPERATOR_VOICE if turn.speaker == "Оператор" else CLIENT_VOICE
    communicate = edge_tts.Communicate(text=turn.text, voice=voice, rate="-8%")
    await communicate.save(str(output_path))


def create_silence(path: Path) -> None:
    run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=16000:cl=mono",
            "-t",
            str(PAUSE_SECONDS),
            "-ar",
            "16000",
            "-ac",
            "1",
            str(path),
        ]
    )


def write_concat_list(paths: list[Path], list_path: Path) -> None:
    lines = [f"file '{path.as_posix()}'" for path in paths]
    list_path.write_text("\n".join(lines), encoding="utf-8")


async def generate_scenario(scenario: Scenario) -> dict[str, object]:
    scenario_dir = BUILD_DIR / scenario.slug
    if scenario_dir.exists():
        shutil.rmtree(scenario_dir)
    scenario_dir.mkdir(parents=True, exist_ok=True)
    silence = scenario_dir / "silence.wav"
    create_silence(silence)

    concat_inputs: list[Path] = []
    transcript: list[dict[str, object]] = []
    cursor = 0.0

    for index, turn in enumerate(scenario.turns):
        mp3_path = scenario_dir / f"{index:02d}.mp3"
        wav_path = scenario_dir / f"{index:02d}.wav"
        await synthesize_turn(turn, mp3_path)
        run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(mp3_path),
                "-ar",
                "16000",
                "-ac",
                "1",
                str(wav_path),
            ]
        )
        duration = ffprobe_duration(wav_path)
        transcript.append(
            {
                "speaker": turn.speaker,
                "start": round(cursor, 2),
                "end": round(cursor + duration, 2),
                "text": turn.text,
            }
        )
        concat_inputs.extend([wav_path, silence])
        cursor += duration + PAUSE_SECONDS

    concat_list = scenario_dir / "concat.txt"
    master_wav = scenario_dir / "master.wav"
    write_concat_list(concat_inputs, concat_list)
    run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            str(master_wav),
        ]
    )

    output_path = OUT_DIR / f"{scenario.slug}.{scenario.output_format}"
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(master_wav),
            "-af",
            PHONE_FILTER,
            *scenario.codec_args,
            str(output_path),
        ]
    )
    duration = ffprobe_duration(output_path)
    sidecar = {
        "file": output_path.name,
        "title": scenario.title,
        "duration_seconds": round(duration, 2),
        "format": scenario.output_format,
        "sample_rate_hz": 8000,
        "phone_like": True,
        "expected_topic": scenario.expected_topic,
        "expected_priority": scenario.expected_priority,
        "expected_compliance_passed": scenario.expected_compliance_passed,
        "notes": scenario.notes,
        "transcript": transcript,
    }
    (OUT_DIR / f"{scenario.slug}.json").write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    txt = "\n".join(f"{item['speaker']}: {item['text']}" for item in transcript)
    (OUT_DIR / f"{scenario.slug}.txt").write_text(txt, encoding="utf-8")
    return sidecar


def write_manifest(items: list[dict[str, object]]) -> None:
    total = round(sum(float(item["duration_seconds"]) for item in items), 2)
    manifest = {
        "generated_by": "scripts/generate_test_audio.py",
        "total_duration_seconds": total,
        "total_duration_minutes": round(total / 60, 2),
        "files": items,
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Generated MTBank Test Audio",
        "",
        f"Total duration: {total} seconds ({total / 60:.2f} minutes)",
        "",
        "| File | Duration | Topic | Priority | Notes |",
        "|---|---:|---|---|---|",
    ]
    for item in items:
        lines.append(
            f"| {item['file']} | {item['duration_seconds']} | "
            f"{item['expected_topic']} | {item['expected_priority']} | {item['notes']} |"
        )
    (OUT_DIR / "manifest.md").write_text("\n".join(lines), encoding="utf-8")


async def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for scenario in SCENARIOS:
        print(f"Generating {scenario.slug}...")
        items.append(await generate_scenario(scenario))
    write_manifest(items)
    total = sum(float(item["duration_seconds"]) for item in items)
    print(f"Generated {len(items)} files, total duration {total:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
