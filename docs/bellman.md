
  Assume that you are a contestant at a quiz program, where you have an opportunity to win substantial amount of money by answering a series of questions. 
  The following are the rules of the contest:

- At each stage of the contest, you are allowed to chose between 3 actions:

    ```
    0 - Pull out and receive reward
    1 - Answer an easy question
    2 - Answer a hard question
    ```
- A reward r_i is received at the end of the contest. "i" is the max question reached before termination. The reward at all preceeding stages is 0.
- The outcome of a question "i" is determined by a probability "p_i". Note that "p_i > p_{i+1}" and "p_i_easy > p_i_hard". The probabilty p_i is not visible to the agent.
- There are a maximum of N questions(N=16). 
- You are allowed a total of 2 wrong answers, after which the game terminates (that is at the 3rd wrong answer). 
- There are a maximum of 16 questions. You receive full reward if the 16th question is reached. 
- If you are terminated after answering an easy question wrong, your reward is 0. 
- If you are terminated after answering a hard question wrong, your reward is reward_i/2. 


  ## Actions:

0 - Pull out and receive reward

1 - Answer an easy question

2 - Answer a hard question
  
  ## Observations:

A list of outcomes of all previous questions. 1 indicates answered correctly, 0 indicates answered wrong, and "" indicates unanswered
[1,1,0,1,0,1,"","","","","","","","","",""]

  ## Rewards:

- Default reward of 0 at each step
- Reward at last step is 0 if terminated on an easy question
- Reward at last step is reward_i/2 if terminated on a hard question
- Reward at last step is reward_i if pulled out after ith question
- Reward at last step is reward_N if Nth question is reached

  ## Termination criteria:

- Nth question is reached (N=16)
- Last action is 0
- Number of wrongly answered questions > 2
