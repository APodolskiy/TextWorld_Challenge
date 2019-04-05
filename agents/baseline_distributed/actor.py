import gym
import spacy
import textworld.gym
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from agents.baseline_distributed.model import LSTM_DQN
from agents.baseline_distributed.utils.data_utils import preprocess
from agents.utils.eps_scheduler import LinearScheduler
class Actor(mp.Process):
    def __init__(self,
                 actor_id: int,
                 eps,
                 request_infos,
                 game_files,
                 config,
                 shared_state,
                 shared_replay_memory,
                 shared_writer):
        super(Actor, self).__init__()
        self.word_vocab = []
        self._load_vocab(vocab_file="./vocab.txt")
        self.EOS_id = self.word2id["</S>"]
        self.SEP_id = self.word2id["SEP"]
        self.writer = shared_writer
        self.game_files = game_files
        self.id = actor_id
        self.request_infos = request_infos
        self.shared_state = shared_state
        self.shared_replay_memory = shared_replay_memory
        self.config = config
        self.nb_epochs = self.config['training']['nb_epochs']
        self.batch_size = self.config['training']['batch_size']
        self.max_nb_steps_per_episode = self.config['training']['max_nb_steps_per_episode']
        self.use_cuda = True and torch.cuda.is_available()
        self.model = LSTM_DQN(config=self.config["model"], word_vocab=self.word_vocab)
        self.nlp = spacy.load('en', disable=['ner', 'parser', 'tagger'])
        self.eps = eps
        self.eps_scheduler = LinearScheduler(self.config["exploration"])
        self.act_steps = 0
        self.episode_steps = 0
        self.num_episodes = 0
        self.NUM_EPOCHS = 10000
        self._episode_started = False
        self.previous_actions: List[str] = []
        self.scores: List[List[int]] = []
        self.dones: List[List[int]] = []
        self.prev_description_id: List = None
        self.prev_command: List = None
    def run(self):
        env_id = textworld.gym.register_games(self.game_files,
                                              self.request_infos,
                                              max_episode_steps=200,
                                              name='training')
        env_id = textworld.gym.make_batch(env_id, batch_size=1, parallel=False)
        env = gym.make(env_id)
        for epoch_no in range(self.NUM_EPOCHS):
            stats = {
                "scores": [],
                "steps": []
            }
            steps = 0
            for _ in tqdm(range(len(self.game_files))):
                obs, infos = env.reset()
                done, score = False, 0
                while not done:
                    action = self.act(obs, infos)
                    obs, score, done, infos = env.step(action)
                    steps += 1
                stats["scores"].append(score)
                stats["steps"].append(steps)
    def act(self, obs: str, infos: Dict[str, List[Any]]) -> str:
        pass
    def get_game_state_info(self, obs: str, infos: Dict[str, List[Any]]) -> torch.Tensor:
        description_tokens = preprocess(infos["description"], "description", tokenizer=self.nlp)
        if len(description_tokens) == 0:
            description_tokens = ["end"]
        description_ids = None
        res = 0
        inventory_tokens = preprocess(infos["inventory"], "inventory", tokenizer=self.nlp)
        inventory_ids = None
        recipe_tokens = preprocess(infos["extra.recipe"], "recipe", tokenizer=self.nlp)
        recipe_ids = None
        feedback_tokens = preprocess(infos["feedback"], "feedback", tokenizer=self.nlp)
        feedback_ids = None
        state_ids = description_ids + inventory_ids + recipe_ids
        input_description = pad_sequences(state_id_list, maxlen=max_len(state_id_list)).astype('int32')
        input_description = to_pt(input_description, self.use_cuda)
        return input_description
