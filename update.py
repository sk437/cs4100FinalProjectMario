for i_episode in range(NUM_EPISODES):
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    distance = 0
    repeat = 0
    prev_action = None
    if (i_episode % 10) == 0:
        datafile.write("\n")
        datafile.write("AVERAGE SCORES PAST 10 EPISODES: " + str(prevTenDistancesSum / 10) + " \n")
        datafile.write("\n")
        prevTenDistancesSum = 0
    if (i_episode == NUM_EPISODES - 10):
        EPS_START = 0
        EPS_END = 0
    for i in count():
        if (repeat == 0):
            action = select_action(state)
            prev_action = action
        else:
            action = prev_action
        repeat += 1
        if (repeat == ACTION_REPEAT): repeat = 0
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        
        last_screen = current_screen
        current_screen = get_screen()
        distance += reward.item()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None
        
        memory.push(state, action, next_state, reward)
        
        state = next_state
        
        optimize_model()
        if done:
            datafile.write("Finished Episode " + str(i_episode) + ", score = " + str(distance) + "\n")
            print("Finished Episode " + str(i_episode) + ", score = " + str(distance))
            prevTenDistancesSum += distance
            break
            
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())