from __future__ import print_function
from time import sleep

import numpy as np
import collections
import codecs
import random
import json

from numpy.core.fromnumeric import shape

from utils.doom import *
from utils.points import *
from constants import *

import math 

actions = [[True, False, False], [False, True, False], [False, False, True]]

ObjectLocation = collections.namedtuple("ObjectLocation", ["x", "y"])
AgentLocation = collections.namedtuple("AgentLocation", ["x", "y", "theta"])


class MultiGoal_GroundingEnv:
    def __init__(self, args):
        """Initializes the environment.

        Args:
          args: dictionary of parameters.
        """
        self.params = args

        # Reading train and test instructions.
        self.train_instructions = self.get_instr(self.params.train_instr_file)
        self.test_instructions = self.get_instr(self.params.test_instr_file)

        if self.params.use_train_instructions:
            self.instructions = self.train_instructions
        else:
            self.instructions = self.test_instructions

        self.word_to_idx = self.get_word_to_idx()
        self.objects, self.object_dict = self.get_all_objects(self.params.all_instr_file)
        self.object_sizes = self.read_size_file(self.params.object_size_file)
        self.objects_info = self.get_objects_info()

        # print(self.object_dict)

    def game_init(self):
        """Starts the doom game engine."""
        game = DoomGame()
        game = set_doom_configuration(game, self.params)
        game.init()
        self.game = game

    def reset(self):
        """Starts a new episode.

        Returns:
           state: A tuple of screen buffer state and instruction.
           reward: Reward at that step.
           is_final: Flag indicating terminal state.
           extra_args: Dictionary of additional arguments/parameters.
        """

        self.game.new_episode()
        self.time = 0
        self.reach_1 = False
        self.reach_2 = False
        self.dist_prev = 0
        self.dist_curr = 0
        self.latter_first = None

        #-----------------------------------------------------
        self.instruction, self.connection_type, instruction_id = self.get_random_instruction()
        seperate_instructions = self.instruction.split(self.connection_type, maxsplit=1)
        self.instruction_1 = seperate_instructions[0][:-1]
        self.instruction_2 = seperate_instructions[1][1:]

        # Retrieve the possible correct objects for the instruction.
        correct_objects_list = self.get_target_objects(instruction_id)

        # Since we fix the number of objects to 5.
        self.correct_location_list = []
        self.correct_location_list.append(np.random.randint(5))
        second_correct_location = np.random.randint(5)
        while second_correct_location == self.correct_location_list[0]:
            second_correct_location = np.random.randint(5)
        self.correct_location_list.append(second_correct_location)

        # Randomly select 2 correct objects from the
        # list of possible correct objects list for the instruction.
        correct_object_id_1 = np.random.choice(correct_objects_list[0])
        correct_object_id_2 = np.random.choice(correct_objects_list[1])
        i = 0
        while correct_object_id_2 == correct_object_id_1:                                                   # infinite loop
            correct_object_id_2 = np.random.choice(correct_objects_list[1])

        chosen_correct_object_1 = [x for x in self.objects_info if
                                 x.name == self.objects[correct_object_id_1]][0]
        chosen_correct_object_2 = [x for x in self.objects_info if
                                 x.name == self.objects[correct_object_id_2]][0]
        
        # Special code to handle 'largest' and 'smallest' since we need to
        # compute sizes for those particular instructions.
        if 'largest' not in self.instruction_1 and 'largest' not in self.instruction_2 \
                and 'smallest' not in self.instruction_1 and 'smallest' not in self.instruction_2:
                object_ids = random.sample([x for x in range(len(self.objects))
                                        if x not in correct_objects_list[0] and x not in correct_objects_list[1]], 3)
        else:
            if ('largest' in self.instruction_1 or 'smallest' in self.instruction_1) and ('largest' in self.instruction_2 or 'smallest' in self.instruction_2):
                object_ids = self.get_candidate_objects_superlative_instr(
                                            chosen_correct_object_1, chosen_correct_object_2, self.instruction_1)[:3] #return 4 but just take only 3
                object_ids = [self.object_dict[x] for x in object_ids]

            elif 'largest' in self.instruction_1 or 'smallest' in self.instruction_1:
                object_ids = self.get_candidate_objects_superlative_instr(
                                            chosen_correct_object_1, chosen_correct_object_2, self.instruction_1)[:3] #return 4 but just take only 3
                object_ids = [self.object_dict[x] for x in object_ids]                             

            elif 'largest' in self.instruction_2 or 'smallest' in self.instruction_2:
                object_ids = self.get_candidate_objects_superlative_instr(
                                            chosen_correct_object_2, chosen_correct_object_1, self.instruction_2)[:3] #return 4 but just take only 3
                object_ids = [self.object_dict[x] for x in object_ids]                             

        assert len(object_ids) == 3

        object_ids.insert(self.correct_location_list[0], correct_object_id_1)
        object_ids.insert(self.correct_location_list[1], correct_object_id_2)
        # print(correct_object_id_1, correct_object_id_2)
        # print(object_ids)
        # Get agent and object spawn locations.
        agent_x_coordinate, agent_y_coordinate, \
            agent_orientation, object_x_coordinates, \
            object_y_coordinates = self.get_agent_and_object_positions()

        self.object_coordinates = [ObjectLocation(x, y) for x, y in
                                   zip(object_x_coordinates,
                                       object_y_coordinates)]

        # Spawn agent.
        spawn_agent(self.game, agent_x_coordinate,
                    agent_y_coordinate, agent_orientation)

        # Spawn objects.
        [spawn_object(self.game, object_id, pos_x, pos_y) for
            object_id, pos_x, pos_y in
            zip(object_ids, object_x_coordinates, object_y_coordinates)]

        screen = self.game.get_state().screen_buffer
        depth = self.game.get_state().depth_buffer
        screen_buf = process_screen(screen, self.params.frame_height,
                                    self.params.frame_width)
        depth_buf = process_depth(depth, self.params.frame_height, 
                                    self.params.frame_width)

        state = (screen_buf, depth_buf, self.instruction)

        '''
        
        #----CALCULATE DISTANCE
        self.agent_x, self.agent_y = get_agent_location(self.game)
        target_location_1 = self.object_coordinates[self.correct_location_list[0]]
        target_location_2 = self.object_coordinates[self.correct_location_list[1]]

        
        if self.connection_type == 'and':
            dist_1 = get_l2_distance(self.agent_x, self.agent_y,
                               target_location_1.x, target_location_1.y)
            dist_2 = get_l2_distance(self.agent_x, self.agent_y,
                               target_location_2.x, target_location_2.y)
            if dist_1 < dist_2:
                self.dist_prev = self.dist_curr = dist_1
                self.latter_first = True
            else:
                self.dist_prev = self.dist_curr = dist_2
                self.latter_first = False

        elif self.connection_type == 'before':
            dist_1 = get_l2_distance(self.agent_x, self.agent_y,
                               target_location_1.x, target_location_1.y)
            self.latter_first = True
            self.dist_prev = self.dist_curr = dist_1
        else:
            dist_2 = get_l2_distance(self.agent_x, self.agent_y,
                               target_location_2.x, target_location_2.y)
            self.latter_first = False
            self.dist_prev = self.dist_curr = dist_2
        #-----------------------------------------
        '''

        reward, shaping_reward = self.get_reward()
        is_final = False
        extra_args = None

        
        #                  
        return state, reward + shaping_reward, is_final # extra_args

    def step(self, action_id):
        """Executes an action in the environment to reach a new state.

        Args:
          action_id: An integer corresponding to the action.

        Returns:
           state: A tuple of screen buffer state and instruction.
           reward: Reward at that step.
           is_final: Flag indicating terminal state.
           extra_args: Dictionary of additional arguments/parameters.
        """
        # Repeat the action for 5 frames.
        if self.params.visualize:
            # Render 5 frames for better visualization.
            for _ in range(5):
                self.game.make_action(actions[int(action_id)], 1)
                # Slowing down the game for better visualization.
                sleep(self.params.sleep)
        else:
            
            self.game.make_action(actions[int(action_id)], 5)

        self.time += 1
        reward, shaping_reward = self.get_reward()
        
        # End the episode if episode limit is reached or
        # agent reached an object.
        is_final = False
        if self.time == self.params.max_episode_length or reward == WRONG_OBJECT_REWARD or (self.reach_1 + self.reach_2 == 2):
            is_final = True
        else:
            is_final = False

        screen = self.game.get_state().screen_buffer
        depth = self.game.get_state().depth_buffer
        depth_buf = process_depth(depth, self.params.frame_height, 
                                    self.params.frame_width)
        screen_buf = process_screen(
            screen, self.params.frame_height, self.params.frame_width)

        state = (screen_buf, depth_buf, self.instruction)

        return state, reward, is_final, (self.reach_1, self.reach_2) # None

    def close(self):
        self.game.close()

    def get_reward(self):
        """Get the reward received by the agent in the last time step."""
        # If agent reached the correct object, reward = +1.
        # If agent reach a wrong object, reward = -0.2.
        # Otherwise, reward = living_reward.

        self.agent_x, self.agent_y = get_agent_location(self.game)
        target_location_1 = self.object_coordinates[self.correct_location_list[0]]
        target_location_2 = self.object_coordinates[self.correct_location_list[1]]
        dist_1 = get_l2_distance(self.agent_x, self.agent_y,
                               target_location_1.x, target_location_1.y)
        dist_2 = get_l2_distance(self.agent_x, self.agent_y,
                               target_location_2.x, target_location_2.y)

        total_reward = 0.
        shaping_reward = 0.
        '''
        
        #----CALCULATE DISTANCE to SHAPING REWARD
        if self.connection_type == 'and':
            if self.latter_first == True and self.reach_1 == False:
                self.dist_curr = dist_1
            else:
                self.dist_curr = dist_2
        elif self.connection_type == 'before':
            if self.reach_1 == False:
                self.dist_curr = dist_1
            else:
                self.dist_curr = dist_2
        else:
            if self.reach_2 == False:
                self.dist_curr = dist_2
            else:
                self.dist_curr = dist_1

        shaping_reward += self.dist_prev - self.dist_curr
        shaping_reward = shaping_reward / 100
        self.dist_prev = self.dist_curr
        #-----------------------------------------
        '''

        if dist_1 <= REWARD_THRESHOLD_DISTANCE:
            # self.dist_prev = self.dist_curr = dist_2
            if self.reach_1 == False:
                if self.connection_type == 'after' and self.reach_2 == False: #collide
                    total_reward += WRONG_OBJECT_REWARD
                    return total_reward, shaping_reward
                elif self.connection_type == 'after' and self.reach_2 == True: #collide
                    total_reward += CORRECT_OBJECT_REWARD
                    self.reach_1 = True
                    # total_reward += CORRECT_OBJECT_REWARD/2 # additional reward when agent reach two goals
                    return total_reward, shaping_reward

                if self.connection_type=='and' or self.connection_type=='before': #collide
                    total_reward += CORRECT_OBJECT_REWARD
                    # if self.connection_type == 'and' and self.reach_2 == True:
                        # total_reward += CORRECT_OBJECT_REWARD/2
                    self.reach_1 = True
            else:
                total_reward += self.params.living_reward
        
        if dist_2 <= REWARD_THRESHOLD_DISTANCE:
            # self.dist_prev = self.dist_curr = dist_1
            if self.reach_2 == False:
                if self.connection_type == 'before' and self.reach_1 == False: #collide
                    total_reward += WRONG_OBJECT_REWARD
                    return total_reward, shaping_reward
                elif self.connection_type == 'before' and self.reach_1 == True: #collide
                    total_reward += CORRECT_OBJECT_REWARD
                    self.reach_2 = True
                    # total_reward += CORRECT_OBJECT_REWARD/2 #additional reward when agent reach two goals
                    return total_reward, shaping_reward
                
                if self.connection_type == 'and' or self.connection_type=='after': #collide
                    total_reward += CORRECT_OBJECT_REWARD
                    # if self.connection_type == 'and' and self.reach_1 == True:
                    #     total_reward += CORRECT_OBJECT_REWARD/2
                    self.reach_2 = True
            else:
                total_reward += self.params.living_reward

        if dist_1 > REWARD_THRESHOLD_DISTANCE and dist_2 > REWARD_THRESHOLD_DISTANCE:
            for i, object_location in enumerate(self.object_coordinates):
                if i == self.correct_location_list[0] or i == self.correct_location_list[1]:
                    continue
                dist = get_l2_distance(self.agent_x, self.agent_y,
                                       object_location.x, object_location.y)
                if dist <= REWARD_THRESHOLD_DISTANCE: #collide
                    total_reward = WRONG_OBJECT_REWARD
                    return total_reward, shaping_reward
            total_reward = self.params.living_reward

        return total_reward, shaping_reward

    def get_agent_and_object_positions(self):
        """Get agent and object positions based on the difficulty
        of the environment.
        """
        object_x_coordinates = []
        object_y_coordinates = []

        if self.params.difficulty == 'easy':
            # Agent location fixed in Easy.
            agent_x_coordinate = 128
            agent_y_coordinate = 512
            agent_orientation = 0

            # Candidate object locations are fixed in Easy.
            object_x_coordinates = [EASY_ENV_OBJECT_X] * 5
            for i in range(-2, 3):
                object_y_coordinates.append(
                    Y_OFFSET + MAP_SIZE_Y/2.0 + OBJECT_Y_DIST * i)

        if self.params.difficulty == 'medium':
            # Agent location fixed in Medium.
            agent_x_coordinate = 128
            agent_y_coordinate = 512
            agent_orientation = 0

            # Generate 5 candidate object locations.
            for i in range(-2, 3):
                object_x_coordinates.append(np.random.randint(
                    MEDIUM_ENV_OBJECT_X_MIN, MEDIUM_ENV_OBJECT_X_MAX))
                object_y_coordinates.append(
                    Y_OFFSET + MAP_SIZE_Y/2.0 + OBJECT_Y_DIST * i)

        if self.params.difficulty == 'hard':
            # Generate 6 random locations: 1 for agent starting position
            # and 5 for candidate objects.
            random_locations = generate_points(HARD_ENV_OBJ_DIST_THRESHOLD,
                                               MAP_SIZE_X - 2*MARGIN,
                                               MAP_SIZE_Y - 2*MARGIN, 6)

            agent_x_coordinate = random_locations[0][0] + X_OFFSET + MARGIN
            agent_y_coordinate = random_locations[0][1] + Y_OFFSET + MARGIN
            agent_orientation = np.random.randint(4)

            for i in range(1, 6):
                object_x_coordinates.append(
                    random_locations[i][0] + X_OFFSET + MARGIN)
                object_y_coordinates.append(
                    random_locations[i][1] + Y_OFFSET + MARGIN)

        return agent_x_coordinate, agent_y_coordinate, agent_orientation, \
            object_x_coordinates, object_y_coordinates

    def get_candidate_objects_superlative_instr(self, correct_object, avoid_object, instruction):
        '''
        Get any possible combination of objects
        and give the maximum size

        SIZE_THRESHOLD refers to the size in terms of number of pixels so that
        atleast there is minimum size difference between two objects for
        instructions with superlative terms (largest and smallest)
        These sizes are stored in ../data/object_sizes.txt
        '''

        instr_contains_color = False
        # instr_contains_color is set if the instruction contains the color
        # attribute (e.g.) "Go to the largest green object".
        # instr_contains_color is True if the instruction doesn't contain the
        # color attribute. (e.g.) "Go to the smallest object"

        instruction_words = instruction.split()
        if len(instruction_words) == 6 and \
                instruction_words[-1] == 'object':
            instr_contains_color = True

        output_objects = []

        # For instructions like "largest red object", the incorrect object
        # set could contain larger objects of other color

        for obj in self.objects_info:
            if obj == avoid_object:
                continue
            if instr_contains_color:
                # first check color attribute
                if correct_object.color != obj.color:
                    output_objects.append(obj)

            if instruction_words[3] == 'largest':
                if correct_object.absolute_size > \
                        obj.absolute_size + SIZE_THRESHOLD:
                    output_objects.append(obj)

            else:
                if correct_object.absolute_size < \
                        obj.absolute_size - SIZE_THRESHOLD:
                    output_objects.append(obj)

        # shuffle the objects and select the top 4
        # randomizing the objects combination
        random.shuffle(output_objects)
        return [x.name for x in output_objects[:4]]

    def get_objects_info(self):
        objects = []
        objects_map = self.objects
        for obj in objects_map:
            split_word = split_object(obj)
            candidate_object = DoomObject(*split_word)
            candidate_object.absolute_size = self.object_sizes[obj]
            objects.append(candidate_object)

        return objects

    # def get_all_objects(self, filename):
    #     objects = []
    #     object_dict = {}
    #     count = 0
    #     instructions = self.get_instr(filename)
    #     for instruction_data in instructions:
    #         object_names_list = instruction_data['targets']
    #         for object_names in object_names_list:
    #             for object_name in object_names:
    #                 if object_name not in objects:
    #                     objects.append(object_name)
    #                     object_dict[object_name] = count
    #                     count += 1

    #     return objects, object_dict

    def get_all_objects(self, filename):
        objects = []
        object_dict = {}
        count = 0
        instructions = self.get_instr(filename)
        for instruction_data in instructions:
            object_names = instruction_data['targets']
            for object_name in object_names:
                if object_name not in objects:
                    objects.append(object_name)
                    object_dict[object_name] = count
                    count += 1

        return objects, object_dict

    def get_target_objects(self, instr_id):
        object_names_list = self.instructions[instr_id]['targets']

        correct_objects_list = []
        correct_objects = []

        for object_names in object_names_list:
            for object_name in object_names:
                correct_objects.append(self.object_dict[object_name])
            correct_objects_list.append(correct_objects)
            correct_objects = []

        return correct_objects_list

    def get_instr(self, filename):
        with open(filename, 'rb') as f:
            instructions = json.load(f)
        return instructions

    def read_size_file(self, filename):
        with codecs.open(filename, 'r') as open_file:
            lines = open_file.readlines()

        object_sizes = {}
        for i, line in enumerate(lines):
            split_line = line.split('\t')
            if split_line[1].strip() in self.objects:
                object_sizes[split_line[1].strip()] = int(split_line[2])

        return object_sizes

    def get_random_instruction(self):
        instruction_id = np.random.randint(len(self.instructions))
        instruction = self.instructions[instruction_id]['instruction']
        connection_type = self.instructions[instruction_id]['connection_type']
        return instruction, connection_type, instruction_id

    def get_word_to_idx(self):
        word_to_idx = {}
        for instruction_data in self.train_instructions:
            instruction = instruction_data['instruction']
            for word in instruction.split(" "):
                if word not in word_to_idx:
                    word_to_idx[word] = len(word_to_idx)
        return word_to_idx
