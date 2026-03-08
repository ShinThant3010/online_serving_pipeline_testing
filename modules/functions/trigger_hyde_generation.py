import threading
import time

import httpx

from modules.utils.load_config import TriggerHydeGenerationConfig

class TriggerHydeGenerationService:

    ### ------------------------------- initialize settings ------------------------------- ###
    def __init__(self, *, config: TriggerHydeGenerationConfig) -> None:
        """Initialize request settings, cooldown policy, and per-student refresh tracking."""
        self._timeout = httpx.Timeout(config.http_timeout_seconds)
        self._refresh_cooldown_seconds = config.refresh_cooldown_seconds
        self._recommendation_api_base_url = config.recommendation_api_base_url
        self._recommendation_path = config.recommendation_path
        self._refresh_lock = threading.Lock()
        self._last_refresh_by_student: dict[str, float] = {}

# ---------------------------------------------------------------------------------------------
# main function: trigger hyde generation of student
# ---------------------------------------------------------------------------------------------
    def trigger_hyde_generation(self, *, student_id: str) -> bool:
        """Fire-and-forget HyDE generation request in a background thread."""
        if not self._is_repeat_call_for_same_student(student_id=student_id):
            # print(f"hyde_generation_request skipped: recent refresh exists for {student_id}")
            return False

        self._start_background_request(student_id=student_id)
        return True


    ### ------- helper: check if the same student is called within config interval ------- ###
    def _is_repeat_call_for_same_student(self, *, student_id: str) -> bool:
        """Determine whether a new HyDE generation request for the same student should be emitted based on cooldown."""
        now = time.monotonic()
        with self._refresh_lock:
            last_refresh = self._last_refresh_by_student.get(student_id)
            if last_refresh is not None and (now - last_refresh) < self._refresh_cooldown_seconds:
                return False
            self._last_refresh_by_student[student_id] = now
            return True


    ### ------------------------------ helper: hyde api call ------------------------------ ###
    def _start_background_request(self, *, student_id: str) -> None:
        """Start a daemon thread that sends the HyDE-generation POST request."""
        def _target() -> None:
            try:
                url = f"{self._recommendation_api_base_url}" + self._recommendation_path.format(
                    student_id=student_id
                )
                with httpx.Client(timeout=self._timeout) as client:
                    response = client.post(
                        url,
                        headers={"accept": "application/json"},
                    )
                print(
                    "hyde_generation_response "
                    f"status={response.status_code} body={response.text}"
                )
            except Exception as exc:
                print(f"hyde_generation_request failed: {exc}")

        # (fire-and-forget) to create a new background worker, so that main request flow does not need to wait
        thread = threading.Thread(
            target=_target,
            daemon=True,
        )
        thread.start()
