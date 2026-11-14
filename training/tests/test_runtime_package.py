"""Tests must exercise the same local Haru code that uv scripts import."""
import os
from pathlib import Path
import subprocess
import sys
import tomllib
import pytest


@pytest.mark.parametrize('from_project_root',[False,True])
def test_haru_dependency_is_editable_and_subprocess_uses_workspace_session(tmp_path,from_project_root):
    project=Path(__file__).resolve().parents[1]
    config=tomllib.loads((project/'pyproject.toml').read_text(encoding='utf-8'))
    assert config['tool']['uv']['sources']['haru-speech']['editable'] is True
    env=os.environ.copy();env.pop('PYTHONPATH',None)
    script='''
import inspect,sys
from pathlib import Path
import haru.session
from haru import DuplexConfig,DuplexSession,HaruModel
assert Path(haru.session.__file__).resolve()==Path(sys.argv[1]).resolve()
assert 'feedback_head' in inspect.signature(haru.session.DuplexSession).parameters
'''
    subprocess.run([sys.executable,'-c',script,str(project.parent/'haru/haru/session.py')],
                   cwd=project.parent if from_project_root else tmp_path,env=env,check=True)
