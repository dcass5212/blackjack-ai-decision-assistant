const DEFAULT_SIMULATION_COUNT = 5000;

// Public API for the UI: evaluate all legal actions from the current game snapshot.
function getActionOdds(gameState, simulations = DEFAULT_SIMULATION_COUNT) {
  const legalActions = getLegalActions(gameState);
  const actionResults = legalActions.map(action => simulateAction(action, gameState, simulations));
  const bestResult = actionResults.reduce((best, result) => (
    result.expectedValue > best.expectedValue ? result : best
  ), actionResults[0]);

  return {
    recommendedAction: bestResult ? bestResult.action : null,
    reason: bestResult ? `${formatActionName(bestResult.action)} has the highest expected value among legal actions.` : 'No legal actions are available.',
    actionResults,
  };
}

// Double down is only included when the live game marks it legal.
function getLegalActions(gameState) {
  if (!gameState || isSimulationBust(gameState.playerHand)) {
    return [];
  }

  const actions = ['hit', 'stand'];

  if (gameState.canDoubleDown) {
    actions.push('doubleDown');
  }

  return actions;
}

// Run repeated random trials for one action and aggregate the outcome rates.
function simulateAction(action, gameState, simulations) {
  const totals = {
    action,
    simulations,
    wins: 0,
    losses: 0,
    pushes: 0,
    busts: 0,
    winRate: 0,
    lossRate: 0,
    pushRate: 0,
    bustRate: 0,
    expectedValue: 0,
  };

  for (let i = 0; i < simulations; i++) {
    const outcome = simulateRound(action, gameState);

    // A player bust is tracked separately for explanation, but still counts as a loss.
    if (outcome === 'busts') {
      totals.busts++;
      totals.losses++;
    } else {
      totals[outcome]++;
    }
  }

  totals.winRate = totals.wins / simulations;
  totals.lossRate = totals.losses / simulations;
  totals.pushRate = totals.pushes / simulations;
  totals.bustRate = totals.busts / simulations;
  totals.expectedValue = calculateExpectedValue(action, totals.winRate, totals.lossRate);

  return totals;
}

// Simulate one possible future round after applying the selected action.
function simulateRound(action, gameState) {
  const simulation = cloneSimulationState(gameState);

  if (action === 'hit') {
    simulation.playerHand.push(drawSimulationCard(simulation.remainingDeck));
    // After choosing hit, the simulated player follows a simple policy: hit until 17+.
    completePlayerTurn(simulation.playerHand, simulation.remainingDeck);
  } else if (action === 'doubleDown') {
    // Double down receives exactly one card, matching the live game rules.
    simulation.playerHand.push(drawSimulationCard(simulation.remainingDeck));
  }

  if (isSimulationBust(simulation.playerHand)) {
    return 'busts';
  }

  playSimulationDealerTurn(simulation.dealerHand, simulation.remainingDeck);
  return resolveSimulationOutcome(simulation.playerHand, simulation.dealerHand);
}

// Clone all mutable data so Monte Carlo trials never affect the actual game.
function cloneSimulationState(gameState) {
  return {
    playerHand: cloneSimulationCards(gameState.playerHand),
    // Phase 3/4 use debug mode: the simulation knows the dealer hole card.
    // A future realistic mode can rebuild dealerHand from dealerUpCard only.
    dealerHand: cloneSimulationCards(gameState.dealerHand),
    remainingDeck: shuffleSimulationDeck(cloneSimulationCards(gameState.remainingDeck)),
  };
}

function cloneSimulationCards(cards) {
  return cards.map(card => ({ r: card.r, s: card.s }));
}

function shuffleSimulationDeck(cards) {
  for (let i = cards.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [cards[i], cards[j]] = [cards[j], cards[i]];
  }

  return cards;
}

function drawSimulationCard(deck) {
  return deck.pop();
}

// Simple stand threshold used only after evaluating the first hit action.
function completePlayerTurn(playerHand, remainingDeck) {
  while (!isSimulationBust(playerHand) && calculateSimulationHandValue(playerHand) < 17) {
    playerHand.push(drawSimulationCard(remainingDeck));
  }
}

// Dealer rule matches the live game: hit below 17, stand on all 17s.
function playSimulationDealerTurn(dealerHand, remainingDeck) {
  while (calculateSimulationHandValue(dealerHand) < 17) {
    dealerHand.push(drawSimulationCard(remainingDeck));
  }
}

function resolveSimulationOutcome(playerHand, dealerHand) {
  const playerScore = calculateSimulationHandValue(playerHand);
  const dealerScore = calculateSimulationHandValue(dealerHand);

  if (dealerScore > 21) return 'wins';
  if (playerScore > dealerScore) return 'wins';
  if (playerScore < dealerScore) return 'losses';
  return 'pushes';
}

function calculateSimulationHandValue(hand) {
  let sum = hand.reduce((total, card) => total + getSimulationCardValue(card.r), 0);
  let aces = hand.filter(card => card.r === 'A').length;

  while (sum > 21 && aces--) {
    sum -= 10;
  }

  return sum;
}

function getSimulationCardValue(rank) {
  if (rank === 'A') return 11;
  if (['J', 'Q', 'K'].includes(rank)) return 10;
  return parseInt(rank);
}

function isSimulationBust(hand) {
  return calculateSimulationHandValue(hand) > 21;
}

// Normalized EV: wins add 1, losses subtract 1, pushes add 0.
// Double down doubles both upside and downside because the bet is doubled.
function calculateExpectedValue(action, winRate, lossRate) {
  const multiplier = action === 'doubleDown' ? 2 : 1;
  return multiplier * (winRate - lossRate);
}

function formatActionName(action) {
  if (action === 'doubleDown') return 'Double down';
  return action.charAt(0).toUpperCase() + action.slice(1);
}
