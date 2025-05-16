package tools;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.*;
import java.util.logging.*;
import cartago.Artifact;
import cartago.OPERATION;
import cartago.OpFeedbackParam;

public class QLearner extends Artifact {

  private Lab lab; // the lab environment that will be learnt
  private int stateCount; // the number of possible states in the lab environment
  private int actionCount; // the number of possible actions in the lab environment
  private HashMap<Integer, double[][]> qTables; // a map for storing the qTables computed for different goals

  private static final Logger LOGGER = Logger.getLogger(QLearner.class.getName());

  public void init(String environmentURL) {

    // the URL of the W3C Thing Description of the lab Thing
    this.lab = new Lab(environmentURL);

    this.stateCount = this.lab.getStateCount();
    LOGGER.info("Initialized with a state space of n=" + stateCount);

    this.actionCount = this.lab.getActionCount();
    LOGGER.info("Initialized with an action space of m=" + actionCount);

    qTables = new HashMap<>();
  }

  /**
   * Computes a Q matrix for the state space and action space of the lab, and
   * against
   * a goal description. For example, the goal description can be of the form
   * [z1level, z2Level],
   * where z1Level is the desired value of the light level in Zone 1 of the lab,
   * and z2Level is the desired value of the light level in Zone 2 of the lab.
   * For exercise 11, the possible goal descriptions are:
   * [0,0], [0,1], [0,2], [0,3],
   * [1,0], [1,1], [1,2], [1,3],
   * [2,0], [2,1], [2,2], [2,3],
   * [3,0], [3,1], [3,2], [3,3].
   *
   * <p>
   * HINT: Use the methods of {@link LearningEnvironment} (implemented in
   * {@link Lab})
   * to interact with the learning environment (here, the lab), e.g., to retrieve
   * the
   * applicable actions, perform an action at the lab during learning etc.
   * </p>
   * 
   * @param goalDescription the desired goal against the which the Q matrix is
   *                        calculated (e.g., [2,3])
   * @param episodesObj     the number of episodes used for calculating the Q
   *                        matrix
   * @param alphaObj        the learning rate with range [0,1].
   * @param gammaObj        the discount factor [0,1]
   * @param epsilonObj      the exploration probability [0,1]
   * @param rewardObj       the reward assigned when reaching the goal state
   **/
  @OPERATION
  public void calculateQ(Object[] goalDescription,
      Object episodesObj,
      Object alphaObj,
      Object gammaObj,
      Object epsilonObj,
      Object rewardObj) {
    // 1) parse parameters
    int episodes = Integer.parseInt(episodesObj.toString());
    double alpha = Double.parseDouble(alphaObj.toString());
    double gamma = Double.parseDouble(gammaObj.toString());
    double epsilon = Double.parseDouble(epsilonObj.toString());
    int goalReward = Integer.parseInt(rewardObj.toString());

    int r1 = Integer.parseInt(goalDescription[0].toString());
    int r2 = Integer.parseInt(goalDescription[1].toString());

    // 2) init Q-table & RNG
    double[][] Q = initializeQTable();
    Random rand = new Random();

    // grab the full state list once
    List<List<Integer>> states = lab.getStateSpace();

    for (int ep = 0; ep < episodes; ep++) {
      // 3) random “kick” to shuffle initial state
      randomizeStartState(rand, 10);

      // 4) read & index the new start state
      int s = lab.readCurrentState();

      int step = 0, maxSteps = 50;
      boolean done = false;
      while (!done && step++ < maxSteps) {
        // 5) ε-greedy pick
        List<Integer> actions = lab.getApplicableActions(s);
        int a;
        if (rand.nextDouble() < epsilon) {
          a = actions.get(rand.nextInt(actions.size()));
        } else {
          // explicit loop to find action with max Q[s][ai]
          a = actions.get(0);
          double bestQ = Q[s][a];
          for (int ai : actions) {
            if (Q[s][ai] > bestQ) {
              bestQ = Q[s][ai];
              a = ai;
            }
          }
        }

        // 6) execute & observe
        lab.performAction(a);
        int sPrime = lab.readCurrentState();
        List<Integer> ns = states.get(sPrime);
        int z1 = ns.get(0), z2 = ns.get(1);
        boolean l1 = ns.get(2) == 1;
        boolean l2 = ns.get(3) == 1;
        boolean b1 = ns.get(4) == 1;
        boolean b2 = ns.get(5) == 1;

        // 7) compute reward
        double match = (z1 == r1 && z2 == r2) ? goalReward : -1.0;
        double energy = -0.5 * ((l1 ? 1 : 0) + (l2 ? 1 : 0))
            - 0.1 * ((b1 ? 1 : 0) + (b2 ? 1 : 0));
        double partial = (3 - Math.abs(z1 - r1)) + (3 - Math.abs(z2 - r2));
        double R = partial * 0.5 + match + energy;

        // 8) Q-update
        double qsa = Q[s][a];
        // find max over next row
        double maxQp = Q[sPrime][0];
        for (int i = 1; i < Q[sPrime].length; i++) {
          if (Q[sPrime][i] > maxQp) {
            maxQp = Q[sPrime][i];
          }
        }
        Q[s][a] = qsa + alpha * (R + gamma * maxQp - qsa);

        // 9) terminal check
        if (z1 == r1 && z2 == r2) {
          done = true;
        } else {
          s = sPrime;
        }
      }
    }

    // store
    qTables.put(Arrays.hashCode(goalDescription), Q);
    LOGGER.info("Finished Q-learning for goal " + Arrays.toString(goalDescription));
    printQTable(Q);
    saveQTable(Q, r1, r2, episodes, alpha, gamma, epsilon);
  }

  /**
   * Returns information about the next best action based on a provided state and
   * the QTable for
   * a goal description. The returned information can be used by agents to invoke
   * an action
   * using a ThingArtifact.
   *
   * @param goalDescription           the desired goal against the which the Q
   *                                  matrix is calculated (e.g., [2,3])
   * @param currentStateDescription   the current state e.g.
   *                                  [2,2,true,false,true,true,2]
   * @param nextBestActionTag         the (returned) semantic annotation of the
   *                                  next best action, e.g.
   *                                  "http://example.org/was#SetZ1Light"
   * @param nextBestActionPayloadTags the (returned) semantic annotations of the
   *                                  payload of the next best action, e.g.
   *                                  [Z1Light]
   * @param nextBestActionPayload     the (returned) payload of the next best
   *                                  action, e.g. true
   **/
  @OPERATION
  public void getActionFromState(Object[] goalDescription,
      Object[] rawTags,
      Object[] rawValues,
      OpFeedbackParam<String> outTag,
      OpFeedbackParam<Object[]> outPayloadTags,
      OpFeedbackParam<Object[]> outPayload) {
    // 1) look up Q‐table
    int key = Arrays.hashCode(goalDescription);
    double[][] Q = qTables.get(key);
    if (Q == null) {
      LOGGER.warning("No Q‐table for goal " + Arrays.toString(goalDescription));
      return;
    }

    // 2) map semantic URI → raw value
    Map<String, Object> m = new HashMap<>();
    for (int i = 0; i < rawTags.length; i++) {
      m.put(rawTags[i].toString(), rawValues[i]);
    }

    // 3) extract exactly our 7 features in the fixed order:
    // [Z1Level,Z2Level, Z1Light,Z2Light, Z1Blinds,Z2Blinds, Sunshine]
    // whether or not the JSON contained Hour or other extra props
    List<Integer> stateVec = List.of(
        // discretize raw lux
        discretizeLightLevel(((Number) m.get("http://example.org/was#Z1Level")).doubleValue()),
        discretizeLightLevel(((Number) m.get("http://example.org/was#Z2Level")).doubleValue()),
        // boolean lights
        ((Boolean) m.get("http://example.org/was#Z1Light")) ? 1 : 0,
        ((Boolean) m.get("http://example.org/was#Z2Light")) ? 1 : 0,
        // boolean blinds
        ((Boolean) m.get("http://example.org/was#Z1Blinds")) ? 1 : 0,
        ((Boolean) m.get("http://example.org/was#Z2Blinds")) ? 1 : 0,
        // discretize sunshine
        discretizeSunshine(((Number) m.get("http://example.org/was#Sunshine")).doubleValue()));

    System.out.println("State vector: " + stateVec);
    // 4) find its row index
    int s = lab.getStateSpace().indexOf(stateVec);
    if (s < 0) {
      LOGGER.warning("State not in space: " + stateVec);
      return;
    }

    // 5) ε‐greedy is done offline; here just pick best among applicable
    List<Integer> valid = lab.getApplicableActions(s);
    int bestA = valid.get(0);
    double bestQ = Q[s][bestA];
    for (int a : valid) {
      if (Q[s][a] > bestQ) {
        bestQ = Q[s][a];
        bestA = a;
      }
    }

    // 6) map bestA → semantic affordance + payload
    String[] affordances = {
        "http://example.org/was#SetZ1Light",
        "http://example.org/was#SetZ2Light",
        "http://example.org/was#SetZ1Blinds",
        "http://example.org/was#SetZ2Blinds"
    };
    int group = bestA / 2;
    boolean value = (bestA % 2 == 1);
    String tag = affordances[group];
    String payloadKey;
    switch (group) {
      case 0:
        payloadKey = "Z1Light";
        break;
      case 1:
        payloadKey = "Z2Light";
        break;
      case 2:
        payloadKey = "Z1Blinds";
        break;
      default:
        payloadKey = "Z2Blinds";
        break;
    }

    // 7) return them
    outTag.set(tag);
    outPayloadTags.set(new Object[] { payloadKey });
    outPayload.set(new Object[] { value });
  }

  /**
   * Tell whether the given raw state matches the goal.
   *
   * @param goalDescription e.g. [2,3]
   * @param rawTags         the returned property tags, e.g. ["http://…#Z1Level",
   *                        …]
   * @param rawValues       the returned values, e.g. [123.4, …, true, …]
   * @param isReached       OUT: true if after discretization z1==goal[0] &&
   *                        z2==goal[1]
   */
  @OPERATION
  public void isGoalReached(Object[] goalDescription,
      Object[] rawTags,
      Object[] rawValues,
      OpFeedbackParam<Boolean> isReached) {
    // build map tag→value
    Map<String, Object> m = new HashMap<>();
    for (int i = 0; i < rawTags.length; i++) {
      m.put(rawTags[i].toString(), rawValues[i]);
    }
    // discretize the two illuminance readings
    double z1raw = ((Number) m.get("http://example.org/was#Z1Level")).doubleValue();
    double z2raw = ((Number) m.get("http://example.org/was#Z2Level")).doubleValue();
    System.out.println("Checking Goal: Raw values : " + z1raw + ", " + z2raw);
    int z1 = discretizeLightLevel(z1raw);
    int z2 = discretizeLightLevel(z2raw);
    // parse goal
    int g1 = Integer.parseInt(goalDescription[0].toString());
    int g2 = Integer.parseInt(goalDescription[1].toString());

    System.out.println("Checking Goal with parsed values: " + z1 + " == " + g1 + " && " + z2 + " == " + g2);
    // compare
    isReached.set(z1 == g1 && z2 == g2);
  }

  // reuse Lab’s discretization thresholds:
  private int discretizeLightLevel(double v) {
    if (v < 50)
      return 0;
    if (v < 100)
      return 1;
    if (v < 300)
      return 2;
    return 3;
  }

  private int discretizeSunshine(double v) {
    if (v < 50)
      return 0;
    if (v < 200)
      return 1;
    if (v < 700)
      return 2;
    return 3;
  }

  /**
   * Print the Q matrix
   *
   * @param qTable the Q matrix
   */
  void printQTable(double[][] qTable) {
    System.out.println("Q matrix");
    for (int i = 0; i < qTable.length; i++) {
      System.out.print("From state " + i + ":  ");
      for (int j = 0; j < qTable[i].length; j++) {
        System.out.printf("%6.2f ", (qTable[i][j]));
      }
      System.out.println();
    }
  }

  /**
   * Initialize a Q matrix
   *
   * @return the Q matrix
   */
  private double[][] initializeQTable() {
    double[][] qTable = new double[this.stateCount][this.actionCount];
    for (int i = 0; i < stateCount; i++) {
      for (int j = 0; j < actionCount; j++) {
        qTable[i][j] = 0.0;
      }
    }
    return qTable;
  }

  /**
   * Save the Q matrix to a CSV file, including state indices.
   *
   * @param Q        the Q matrix
   * @param r1       the first goal value
   * @param r2       the second goal value
   * @param episodes the number of episodes run
   * @param alpha    the learning rate
   * @param gamma    the discount factor
   * @param epsilon  the exploration probability
   */
  private void saveQTable(double[][] Q,
      int r1,
      int r2,
      int episodes,
      double alpha,
      double gamma,
      double epsilon) {
    // construct filename as before
    String filename = String.format(
        "qtable_%d_%d_episode%d_alpha%f_gamma%f_epsilon%f.csv",
        r1, r2, episodes, alpha, gamma, epsilon);

    try (PrintWriter writer = new PrintWriter(new File(filename))) {
      // 1) write header: state, a0, a1, ..., aN
      writer.println("state," +
          IntStream.range(0, actionCount)
              .mapToObj(i -> "a" + i)
              .collect(Collectors.joining(",")));

      // 2) write each row: stateIndex, Q[state][0], Q[state][1], ...
      for (int i = 0; i < Q.length; i++) {
        String row = i + "," + // <-- prepend the state index
            Arrays.stream(Q[i])
                .mapToObj(v -> String.format("%.4f", v))
                .collect(Collectors.joining(","));
        writer.println(row);
      }

      LOGGER.info("Saved Q-table (with state indices) to “" + filename + "”");
    } catch (IOException e) {
      LOGGER.severe("Failed to write Q-table file: " + e.getMessage());
    }
  }

  private void randomizeStartState(Random rand, int maxKicks) {
    // pick a random number of kicks (1..maxKicks)
    int kicks = rand.nextInt(maxKicks) + 1;
    for (int i = 0; i < kicks; i++) {
      // read current row‐index, get its valid actions, then perform one at random
      int cur = lab.readCurrentState();
      List<Integer> valid = lab.getApplicableActions(cur);
      lab.performAction(valid.get(rand.nextInt(valid.size())));
    }
  }
}
