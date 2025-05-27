# chinese_checkers_ai
AI for Chinese Checkers board game

# version 1 notes

We are using NEAT to learn, I ran it through the night letting it play is self and here is what I learn, it indeed did learn to move it pieces forward, but mainly just one piece, and it doesn't really know about jumps. So the game ends up bein it moving a few piece 1 forward and just taking one to the end and continue playing with it. Also training is super slow, only did Input Output NN (732 paramters) with 20 matches of 2-6 players per generation with 50 max moves a generation and the whole night it completed 36 generations. It takes 1-10 seconds per round, we do multi thread this but still expected much faster results with our small model. Looking at what takes so long, it takes 0.001 - 0.003 seconds to generate all possible next moves and 0.002 - 0.01 to convert to tensor and 0.02 - 0.08 to evaluate a moves quality. And takes 0.5 to 1.2 seconds to choose best move from all possible moves in a round. So that makes 3-7.2 seconds per round max (around what we are getting with some fluxation). So the bottle next the network itself and a very little bit tensor conversion. We can instead of evaluting each move with indivual call make one back for a all moves for a round. And maybe also store the boards state as a tensor so no conversion is needed.

After switching to batch evalution for best move and enabling no_grad and lowering threads from 8 to 4 we are getting 1-5 seconds per round, mag 2 improvement in speed, but still slow. First off the way we are storing state at the moment seems like there is a lot of redudancies, not to meantion just creating the state tensor takes 0.006*50*6 ~= 1.8 seconds (a substantial amount of runtime since 1-5 seconds is the totla round time). The actual calculating of the moves isn't substatial enough at the moment with 0.003*6 ~= 0.02.

Currently we create matches with 2-6 players each with there own network with 20 matches a generation(so 40-120 models a generation), alternative generation suggestions.

1. Tourmentant style: maybe v6 first round then v5 second and so on until v2

versus:  v2 -> v3 -> v4 -> v5  -> v6
players: 2  -> 6  -> 24 -> 120 -> 720
matches: 1  -> 3  -> 9  -> 33  -> 153

with this we can test more players in less matches and have a balanced ability to play any amount of players (generialized)?

or always opposed tourmentant

versus: v2 -> v4 -> v6
players: 2 -> 8  -> 48
matches: 1 -> 3  -> 11

this tourmentant style may reflects real live play more as you usualy play with someone always directly oppsing you, plus can scale up much more (run multiple tourmenants a generation or longer tourmenants)

Ok let's go with tourmentant generations. Now to rethink how state is stored to reduced redundancy. The board dimensions stays the same always, only the pieces move. each piece's cooridants can is 2 values (will be using euclidies). so (10(pieces)+1(current player+is_red))*6(players)*2(coordinates)= 132  (from 732) tensor dimensions = (6, 11, 2)

where [0] is the top player with [1] right from him and so on until [5]

# version 2 notes

rewrote the whole program so that the modal uses 132 elements to represent state and is
store in torch state (so no convertion needed) (althoug regert using the getattr and dataclass convention, should just have done plan python objects or dataclass as not dataclass).

I've implemented deep q learning and no matter what I do it doesn't seem to work. I think it is how I'm actually reinforcing the network that is the issue, but could also be something of the game implementation itself? So first we must make sure when the game is being simulated in the training that is behaving as expected, so implementing it visually could help (not with performance though). Then after we are sure that is is behacing correctly we can relook the q reinforcment or other algorithm

Ok I've setup a visual deep q simulator (just running matches no epochs or actual evolution, but uses the train network ouputed by the console deepq.network). So will need to reevaluate how the reinforcement is done

So something is difently wrong with the rotation or scoring, also my perfectionism want me to recode all this again, but beter. Or I'm just not in the mood to critical think the ai. I'm going to fix the scoring and see how I feel. 

Ok found issue, wasn't the rotation but rather the selection of player key on score evaultation. How I wrote the board code is so tedius to go through. and the reliance on tensor as the state sounded good but makes for hard usabilty. But after already recoding this 2 times I think I can create a 3rd that is high level enough to make usability great with good performance. But I fear I'll stay in a loop of just recoding without resolving the actual issue I have which is I don't know how to create a q function that works for the program. Before retrying I'll create a evolution trainer again, see how that fairs

So used cursor to generate some ai slop and do vibe coding for the first time since I was having trouble. I have a mixed response, it help improved the training for evolution.py but struggled with deepq with dynamic action space, deepq2 was wholly generated by the ai and it uses the transformer architecture which is really slow on this laptop so unsure if that even works, the heuristics controller worked the best until the ai broke it's functionality

Through all this I've had some ideas for a beter v3 and will be working on that

#version 3 notes

so rewrote the board model in a way I think I won't change, it is fast and flexable, with a position having a euclid position for drawing and a int tuble for a key for set lookup. I used the ai to mostly just do auto complete, except fot the to_tensor method in board which I used cursor to generate, still needed to do some touchups myself, but cursor did legitamitly give a good idea of what and how to do a to_tensor.

Go a deterministic*(added randomize to make less deterministic) heurisitic controller which I wrote myself but took some ideas from the heurisitc controller cursor wrote for v2 that never worked, this controller is able to win me always, wanna still write a ai that learns instead of having a algorithm. But haven't done q learning before and tried learning it via this project but did not figure it out, so going to do a q learning (maybe also a policy gradient too) and then come back to this