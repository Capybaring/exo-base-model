"""The single-command launcher forwards every flag and stops on a failed run."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('first_exit, expected_names', [(0, ['NP', 'BE']), (7, ['NP'])])
def test_np_be_launcher(tmp_path, first_exit, expected_names):
    # Place the launcher in a fake checkout so the test never writes real results.
    root = tmp_path/'repo with spaces'
    scripts = root/'scripts'
    scripts.mkdir(parents=True)
    script = scripts/'run_np_be_patch_lift.sh'
    script.write_bytes((ROOT/'scripts'/script.name).read_bytes())
    recorder = tmp_path/'record.py'
    recorder.write_text('''import json, os, sys
from pathlib import Path
with open(os.environ['CALLS'], 'a') as stream:
    stream.write(json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd()})+'\\n')
sys.exit(int(os.environ['FIRST_EXIT']))
''')
    fake_python = tmp_path/'python'
    fake_python.write_text('#!/bin/sh\nexec "$TEST_PYTHON" "$TEST_RECORDER" "$@"\n')
    fake_python.chmod(0o755)
    calls = tmp_path/'calls.jsonl'
    env = dict(os.environ, PYTHON_BIN=str(fake_python), TEST_PYTHON=sys.executable,
               TEST_RECORDER=str(recorder), CALLS=str(calls), FIRST_EXIT=str(first_exit), DEVICE='cpu')
    result = subprocess.run(['bash', str(script)], cwd=tmp_path, env=env,
                            capture_output=True, text=True)
    assert result.returncode == first_exit, result.stderr
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(commands) == len(expected_names)
    save_roots = []
    for call, name in zip(commands, expected_names):
        assert call['cwd'] == str(root)
        assert call['argv'][:3] == ['-u', '-m', 'exobase.train']
        flags = dict(zip(call['argv'][3::2], call['argv'][4::2]))
        assert flags['--config'] == str(root/'configs'/f'{name}.json')
        assert flags['--sample-stride'] == '43200'
        assert flags['--patch-seconds'] == flags['--patch-stride'] == '43200'
        assert flags['--history'] == '604800' and flags['--forecast'] == '86400'
        assert flags['--batch-size'] == '64' and flags['--device'] == 'cpu'
        assert flags['--lag-estimator'] == 'fft' and flags['--refiner-mode'] == 'global'
        destination = Path(flags['--save-path'])
        assert destination.is_absolute() and destination.is_dir()
        assert destination.parts[-2:] == (name, 'patch-lift')
        save_roots.append(destination.parent.parent)
    assert len(set(save_roots)) == 1
