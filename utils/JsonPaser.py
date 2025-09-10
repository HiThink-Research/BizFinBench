import json
import re

class JsonPaser:
    _code_fence_json = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
    _json_prefixed    = re.compile(r"\bjson\s*({[\s\S]*})", re.IGNORECASE)

    def __init__(self):
        pass

    def extract_json_from_text(self, text):
        if not isinstance(text, str):
            return None

        # 1) 直接从代码块、json{...}里收集候选（不修改原文）
        candidates = []
        candidates += [m.group(1) for m in self._code_fence_json.finditer(text)]
        candidates += [m.group(1) for m in self._json_prefixed.finditer(text)]

        # 2) 用大括号配对扫描整段文本，额外收集 { ... } 片段
        candidates += self._brace_objects(text)

        # 3) 去重 & 按长度从短到长尝试（短的更“纯”）
        seen, uniq = set(), []
        for c in candidates:
            c = c.strip()
            if c and c not in seen:
                seen.add(c)
                uniq.append(c)
        uniq.sort(key=len)
        
        for cand in uniq:
            fixed = self._light_fix(cand)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                continue
        return None

    # ---- helpers ----
    def _brace_objects(self, s):
        """栈扫描：提取所有平衡的 { ... } 顶层片段（不关心 <think> 存在与否）"""
        res, stack = [], []
        start, in_str, esc = None, False, False
        for i, ch in enumerate(s):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    stack.append("{")
                    if start is None:
                        start = i
                elif ch == "}":
                    if stack:
                        stack.pop()
                        if not stack and start is not None:
                            res.append(s[start:i+1])
                            start = None
        return res

    def _light_fix(self, js):
        """温和修复：奇怪引号、尾逗号、字符串内裸换行"""
        # 引号标准化
        # js = js.replace("“", '"').replace("”", '"').replace("„", '"').replace("‟", '"').replace("＂", '"')
        # js = js.replace("’", "'").replace("‘", "'")
        # 删除 ,} 和 ,] 尾逗号
        js = re.sub(r",(\s*[}\]])", r"\1", js)
        # 仅在字符串内部把裸换行替换为 \n
        return self._escape_newlines_in_strings(js)

    def _escape_newlines_in_strings(self, s):
        out, in_str, esc = [], False, False
        for ch in s:
            if in_str:
                if esc:
                    out.append(ch); esc = False
                else:
                    if ch == "\\":
                        out.append(ch); esc = True
                    elif ch == '"':
                        out.append(ch); in_str = False
                    elif ch in ("\n", "\r"):
                        out.append("\\n")
                    else:
                        out.append(ch)
            else:
                if ch == '"':
                    in_str = True
                out.append(ch)
        return "".join(out)