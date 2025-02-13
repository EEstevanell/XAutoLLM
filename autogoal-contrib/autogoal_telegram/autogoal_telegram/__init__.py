import statistics
import time
import textwrap
import re

from autogoal.search import Logger
from telegram import Bot

class TelegramLogger(Logger):
    def __init__(self, token, channel: str = None, name="", objectives=None):
        self.name = name
        self.channel = channel
        self.last_time = time.time()
        self.last_time_other = time.time()
        self.bot = Bot(token=token)
        self.progress = 0

        self.errors = 0
        self.timeout_errors = 0
        self.cuda_errors = 0
        self.memory_errors = 0

        self.generations = 1
        self.current_generation = 1
        self.bests = []
        self.bests_pipelines = []
        self.current = ""
        self.message_index = 0
        self.objectives = objectives

        try:
            self.message = self.bot.send_message(
                chat_id=self.channel,
                text=f"<b>{self.name}</b>\nStarting...",
                parse_mode="HTML",
            )
            self.bot.pin_chat_message(chat_id=self.channel, message_id=self.message.message_id)
        except Exception as e:
            print(f"Exception during initialization: {e}")
            self.message = None

        self.last_solution_message = None

    def begin(self, generations, pop_size):
        self.generations = generations
        self._safe_call(self._send_status, "begin")

    def update_best(
        self,
        solution,
        fn,
        new_best_solutions,
        best_solutions,
        new_best_fns,
        best_fns,
        new_dominated_solutions,
    ):
        self.bests = new_best_fns
        self.bests_pipelines = new_best_solutions
        self._safe_call(self._send_status, "update_best")

    def end(self, best_solutions, best_fns):
        self.bests_pipelines = best_solutions
        self.bests = best_fns
        self._safe_call(self._send_status, "end")

    def sample_solution(self, solution):
        text = f"""
        Evaluating pipeline:
        Pipeline: <code>{repr(solution)}</code>
        """
        self.last_solution_message = self._safe_call(
            self._send_new, "sample_solution", textwrap.dedent(text)
        )

    def error(self, e: Exception, solution):
        text = f"""
        Error:
        <u>{e}</u>
        """
        self.errors += 1
        if re.search("time for execution", str(e).lower()):
            self.timeout_errors += 1

        if re.search("cuda out of memory", str(e).lower()):
            self.cuda_errors += 1

        if self.last_solution_message and hasattr(self.last_solution_message, 'html_text'):
            previous_message = self.last_solution_message.html_text
            self._safe_call(
                self._send_update,
                "error",
                textwrap.dedent(previous_message + text),
                self.last_solution_message,
            )
        else:
            self._safe_call(self._send_new, "error", textwrap.dedent(text))

    def eval_solution(self, solution, fitness, observations):
        self.progress += 1
        self._safe_call(self._send_status, "eval_solution")

        if "inf" in str(fitness).lower():
            return

        time_obs_message = ""
        other_obs_message = ""

        if observations:
            try:
                train_m_time = statistics.mean(observations["time"]["train"])
                valid_m_time = statistics.mean(observations["time"]["valid"])
                train_m_time_value = (
                    (train_m_time, "seconds")
                    if train_m_time < 12000
                    else (train_m_time / 60, "minutes")
                )
                valid_m_time_value = (
                    (valid_m_time, "seconds")
                    if valid_m_time < 12000
                    else (valid_m_time / 60, "minutes")
                )
                time_obs_message = f"train mean time: {train_m_time_value[0]} {train_m_time_value[1]}\nvalid mean time: {valid_m_time_value[0]} {valid_m_time_value[1]}"
            except Exception as e:
                print(f"Exception processing time observations: {e}")

            other_obs_list = []
            try:
                for label, value in observations.items():
                    if label not in ('time', 'resource_stats'):
                        other_obs_list.append(f"Mean {label}: {value}")
                other_obs_message = "\n".join(other_obs_list)
            except Exception as e:
                print(f"Exception processing other observations: {e}")

        text = f"""
        Success:
        <u>{fitness}</u>
        Observations:
        {time_obs_message}
        {other_obs_message}
        """

        if self.last_solution_message and hasattr(self.last_solution_message, 'html_text'):
            previous_message = self.last_solution_message.html_text
            self._safe_call(
                self._send_update,
                "eval_solution",
                textwrap.dedent(previous_message + text),
                self.last_solution_message,
            )
        else:
            self._safe_call(self._send_new, "eval_solution", textwrap.dedent(text))

    def _send_update(self, text, message=None):
        if not self.channel or not message:
            return

        if time.time() - self.last_time_other < 2:
            time.sleep(2)

        self.last_time_other = time.time()

        try:
            message.edit_text(text=f"{text}", parse_mode="HTML")
        except Exception as e:
            print(f"Exception in _send_update: {e}")

    def _send_new(self, text):
        if not self.channel:
            return

        if time.time() - self.last_time_other < 2:
            time.sleep(2)

        self.last_time_other = time.time()
        self.message_index += 1
        try:
            return self.bot.send_message(
                chat_id=self.channel,
                text=f"<b>{self.name} [{self.message_index}]:</b>\n\n{text}",
                parse_mode="HTML",
            )
        except Exception as e:
            print(f"Exception in _send_new: {e}")

    def _send_status(self):
        if not self.channel:
            return

        if time.time() - self.last_time < 5:
            time.sleep(5)

        self.last_time = time.time()
        pareto_front = "["
        for i in range(len(self.bests_pipelines)):
            eval_text = ""
            if not self.objectives:
                eval_text = f"({self.bests[i][0]}, {self.bests[i][1]})"
            else:
                initial = True
                for j in range(len(self.objectives)):
                    obj_name = ""
                    unit = ""

                    if not initial:
                        eval_text += ", "
                    else:
                        initial = False

                    if isinstance(self.objectives[j], tuple):
                        obj_name = self.objectives[j][0]
                        unit = self.objectives[j][1]
                    elif isinstance(self.objectives[j], str):
                        obj_name = self.objectives[j]

                    try:
                        eval_value = self.bests[i][j]
                    except IndexError:
                        eval_value = "N/A"

                    eval_text += f"{obj_name}=<code>{eval_value}"
                    if unit:
                        eval_text += f" {unit}"
                    eval_text += "</code>"

            pareto_front += "\n---------------\n<code>"
            pareto_front += repr(self.bests_pipelines[i])
            pareto_front += "</code>\n"
            pareto_front += eval_text
            pareto_front += "\n---------------\n"
        pareto_front += "]"

        text = textwrap.dedent(
            f"""
            <b>{self.name}</b>

            Iterations: `{self.progress}/{self.generations}`

            ------ ERRORS ------
            Timeouts: `{self.timeout_errors}`
            CUDA: `{self.cuda_errors}`
            Other: `{self.errors - self.timeout_errors - self.cuda_errors}`
            Total Errors: `{self.errors}`

            ------ PARETO FRONT ------
            `{pareto_front}`
            """
        )
        if self.message:
            try:
                self.message.edit_text(text=text, parse_mode="HTML")
            except Exception as e:
                print(f"Exception in _send_status: {e}")

    def _safe_call(self, func, method_name, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Exception in {method_name}: {e}")
