import asyncio
import threading
import time

import httpx

from modules.utils.load_config import TriggerHydeGenerationConfig

class TriggerHydeGenerationService:
    def __init__(self, *, config: TriggerHydeGenerationConfig) -> None:
        self._timeout = httpx.Timeout(config.http_timeout_seconds)
        self._refresh_cooldown_seconds = config.refresh_cooldown_seconds
        self._recommendation_api_base_url = config.recommendation_api_base_url
        self._recommendation_path = config.recommendation_path
        self._refresh_lock = threading.Lock()
        self._last_refresh_by_student: dict[str, float] = {}

    def trigger_hyde_generation(self, *, student_id: str) -> bool:
        """Fire-and-forget HyDE generation request in a background thread."""
        if not self._should_emit_refresh(student_id=student_id):
            print(f"hyde_generation_request skipped: recent refresh exists for {student_id}")
            return False

        self._start_background_request(student_id=student_id)
        return True

    def _start_background_request(self, *, student_id: str) -> None:
        thread = threading.Thread(
            target=self._run_hyde_generation,
            kwargs={"student_id": student_id},
            daemon=True,
        )
        thread.start()

    def _should_emit_refresh(self, *, student_id: str) -> bool:
        """Determine whether a new HyDE generation request should be emitted based on cooldown."""
        now = time.monotonic()
        with self._refresh_lock:
            last_refresh = self._last_refresh_by_student.get(student_id)
            if last_refresh is not None and (now - last_refresh) < self._refresh_cooldown_seconds:
                return False
            self._last_refresh_by_student[student_id] = now
            return True

    def _run_hyde_generation(self, *, student_id: str) -> None:
        try:
            asyncio.run(self._post_hyde_generation(student_id=student_id))
        except Exception as exc:  
            print(f"hyde_generation_request failed: {exc}")

    async def _post_hyde_generation(self, *, student_id: str) -> None:
        url = self._build_request_url(student_id=student_id)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                url,
                headers={"accept": "application/json"},
            )
        print(
            "hyde_generation_response "
            f"status={response.status_code} body={response.text}"
        )

    def _build_request_url(self, *, student_id: str) -> str:
        return f"{self._recommendation_api_base_url}" + self._recommendation_path.format(
            student_id=student_id
        )
