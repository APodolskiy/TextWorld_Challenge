import pytest

from scripts.sample_games import get_skill_count, acquire_skills, count_skills


@pytest.fixture()
def game_name():
    return 'tw-play-recipe2+go1+open1+cook3-12-XJKDASADA'


@pytest.fixture()
def skills(game_name):
    return game_name.split('-')[2]


def test_get_skill_count(skills):
    skills = ['goal3', 'goal', 'goal32', 'recipe2', 'rec2ipe3', 'co2ok']
    true_cnt = [3, 1, 2, 2, 3, 1]
    skill_cnt = [get_skill_count(skill) for skill in skills]
    assert true_cnt == skill_cnt


def test_acquire_skills(game_name):
    true_skills = {
        'recipe': 2,
        'go': 1,
        'open': 1,
        'cook': 3
    }

    acquired_skills = acquire_skills(game_name)
    assert true_skills == acquired_skills


def test_count_skills(game_name):
    true_cnt = 7
    skills_cnt = count_skills(game_name)
    assert true_cnt == skills_cnt
