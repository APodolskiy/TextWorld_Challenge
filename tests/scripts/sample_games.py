import pytest

from scripts.sample_games import get_skill_count, acquire_skills, count_skills


@pytest.fixture()
def game_name():
    return 'tw-play-recipe2+go12+open+cook3-12-XJKDASADA'


@pytest.fixture()
def skills(game_name):
    return game_name.split('-')[2].split('+')


def test_get_skill_count(skills):
    true_cnt = [2, 12, 1, 3]
    skill_cnt = [get_skill_count(skill) for skill in skills]
    assert true_cnt == skill_cnt


def test_acquire_skills(game_name):
    true_skills = {
        'recipe': 2,
        'go': 12,
        'open': 1,
        'cook': 3
    }

    acquired_skills = acquire_skills(game_name)
    assert true_skills == acquired_skills


def test_count_skills(game_name):
    true_cnt = 18
    skills_cnt = count_skills(game_name)
    assert true_cnt == skills_cnt
