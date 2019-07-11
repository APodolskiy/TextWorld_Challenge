from collections import namedtuple


Transition = namedtuple('Transition', ('description_id_list',
                                       'command',
                                       'reward',
                                       'mask',
                                       'done',
                                       'next_description_id_list',
                                       'next_commands'))

SimpleTransition = namedtuple('SimpleTransition', ('description_ids',
                                                   'command_ids',
                                                   'reward',
                                                   'done',
                                                   'next_description_ids',
                                                   'next_command_ids'))
