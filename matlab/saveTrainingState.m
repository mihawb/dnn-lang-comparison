function stop = saveTrainingState(info)
    global training_state
    training_state = [training_state info];
    stop = false;
end