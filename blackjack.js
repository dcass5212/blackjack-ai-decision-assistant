// Game constants
const SUITS = ['&spades;', '&hearts;', '&diams;', '&clubs;'];
const RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'];

// Game state
let deck = [];
let playerHand = [];
let dealerHand = [];
let balance = 500;
let bet = 0;
let pendingBet = 0;
let wins = 0;
let losses = 0;
let roundActive = false;

const TEST_HAND_BET = 25;
const TEST_HANDS = {
  hard16: {
    label: 'Hard 16 vs dealer 10',
    playerHand: [makeCard('10', '&spades;'), makeCard('6', '&hearts;')],
    dealerHand: [makeCard('10', '&clubs;'), makeCard('7', '&diams;')],
  },
  soft18: {
    label: 'Soft 18 vs dealer 9',
    playerHand: [makeCard('A', '&hearts;'), makeCard('7', '&clubs;')],
    dealerHand: [makeCard('9', '&spades;'), makeCard('6', '&diams;')],
  },
  double11: {
    label: '11 vs dealer 6',
    playerHand: [makeCard('5', '&clubs;'), makeCard('6', '&spades;')],
    dealerHand: [makeCard('6', '&hearts;'), makeCard('10', '&diams;')],
  },
  stand20: {
    label: '20 vs dealer 10',
    playerHand: [makeCard('K', '&hearts;'), makeCard('Q', '&clubs;')],
    dealerHand: [makeCard('10', '&hearts;'), makeCard('8', '&spades;')],
  },
};

// Deck creation and shuffling
function createDeck() {
  const newDeck = [];

  for (let s of SUITS) {
    for (let r of RANKS) {
      newDeck.push({ r, s });
    }
  }

  return newDeck;
}

function shuffleDeck(deckToShuffle) {
  for (let i = deckToShuffle.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [deckToShuffle[i], deckToShuffle[j]] = [deckToShuffle[j], deckToShuffle[i]];
  }

  return deckToShuffle;
}

function resetDeck() {
  deck = shuffleDeck(createDeck());
}

// Card drawing
function drawCard() {
  return deck.pop();
}

function makeCard(rank, suit) {
  return { r: rank, s: suit };
}

// Hand scoring
function cardValue(rank) {
  if (rank === 'A') return 11;
  if (['J', 'Q', 'K'].includes(rank)) return 10;
  return parseInt(rank);
}

function calculateHandValue(hand) {
  let sum = hand.reduce((total, card) => total + cardValue(card.r), 0);
  let aces = hand.filter(card => card.r === 'A').length;

  while (sum > 21 && aces--) {
    sum -= 10;
  }

  return sum;
}

// Blackjack and bust helpers
function isBust(hand) {
  return calculateHandValue(hand) > 21;
}

function isBlackjack(hand) {
  return hand.length === 2 && calculateHandValue(hand) === 21;
}

// AI game-state snapshot
function cloneCard(card) {
  return card ? { r: card.r, s: card.s } : null;
}

function cloneHand(hand) {
  return hand.map(cloneCard);
}

function getCurrentGameState() {
  return {
    playerHand: cloneHand(playerHand),
    dealerUpCard: cloneCard(dealerHand[0]),
    dealerHand: cloneHand(dealerHand),
    remainingDeck: cloneHand(deck),
    currentBet: bet,
    balance,
    canDoubleDown: bet > 0 && balance >= bet && playerHand.length === 2,
  };
}

// Balance and betting logic
function addBet(amount) {
  if (pendingBet + amount > balance) {
    msg('Not enough balance!');
    return;
  }

  pendingBet += amount;
  updateUI();
}

function clearBet() {
  pendingBet = 0;
  msg('');
  updateUI();
}

function applyBet() {
  bet = pendingBet;
  pendingBet = 0;
  balance -= bet;
}

function payBlackjack() {
  balance += Math.floor(bet * 2.5);
  wins++;
  showResult('BLACKJACK! +$' + Math.floor(bet * 1.5), 'win');
}

function payWin(message) {
  balance += bet * 2;
  wins++;
  showResult(message + ' +$' + bet, 'win');
}

function returnBet() {
  balance += bet;
  showResult('PUSH &mdash; Bet returned', 'push');
}

function recordLoss(message) {
  losses++;
  showResult(message + ' -$' + bet, 'lose');
}

function finishBet() {
  document.getElementById('bal').textContent = balance;
  document.getElementById('cur-bet').textContent = 0;
  bet = 0;

  if (balance === 0) {
    msg('Out of chips! Reset to play again.');
  }
}

// Dealer turn logic
function playDealerTurn() {
  let dealerScore = calculateHandValue(dealerHand);

  while (dealerScore < 17) {
    dealerHand.push(drawCard());
    dealerScore = calculateHandValue(dealerHand);
  }

  resolveRound();
}

function dealerPlay() {
  playDealerTurn();
}

// Round resolution
function dealHand() {
  if (pendingBet === 0) {
    msg('Place a bet first!');
    return;
  }

  if (deck.length < 15) {
    resetDeck();
  }

  applyBet();
  roundActive = true;
  playerHand = [drawCard(), drawCard()];
  dealerHand = [drawCard(), drawCard()];

  clearResult();
  msg('');
  updateUI(false);
  setPhase('play');
  document.getElementById('double-btn').disabled = balance < bet;

  if (isBlackjack(playerHand)) {
    isBlackjack(dealerHand) ? endRound('push') : endRound('blackjack');
  }
}

function loadTestHand(testHandId) {
  const testHand = TEST_HANDS[testHandId];

  if (!testHand) {
    msg('Test hand not found.');
    return;
  }

  if (balance < TEST_HAND_BET) {
    msg('Not enough balance for a test hand.');
    return;
  }

  bet = TEST_HAND_BET;
  pendingBet = 0;
  balance -= bet;
  playerHand = cloneHand(testHand.playerHand);
  dealerHand = cloneHand(testHand.dealerHand);
  deck = buildDeckWithoutCards([...playerHand, ...dealerHand]);
  roundActive = true;

  clearResult();
  msg(`Loaded test hand: ${testHand.label}`);
  updateUI(false);
  setPhase('play');
  document.getElementById('double-btn').disabled = balance < bet || playerHand.length !== 2;
}

function buildDeckWithoutCards(cardsToRemove) {
  const testDeck = createDeck();

  for (let card of cardsToRemove) {
    removeCardFromDeck(testDeck, card);
  }

  return shuffleDeck(testDeck);
}

function removeCardFromDeck(deckToUpdate, cardToRemove) {
  const cardIndex = deckToUpdate.findIndex(card => cardsMatch(card, cardToRemove));

  if (cardIndex !== -1) {
    deckToUpdate.splice(cardIndex, 1);
  }
}

function cardsMatch(firstCard, secondCard) {
  return firstCard.r === secondCard.r && firstCard.s === secondCard.s;
}

function hit() {
  playerHand.push(drawCard());
  updateUI(false);

  if (isBust(playerHand)) {
    endRound('bust');
  } else if (calculateHandValue(playerHand) === 21) {
    stand();
  }
}

function stand() {
  playDealerTurn();
}

function doubleDown() {
  if (balance < bet) {
    msg('Not enough balance!');
    return;
  }

  balance -= bet;
  bet *= 2;
  playerHand.push(drawCard());
  updateUI(false);

  isBust(playerHand) ? endRound('bust') : playDealerTurn();
}

function resolveRound() {
  const dealerScore = calculateHandValue(dealerHand);
  const playerScore = calculateHandValue(playerHand);

  updateUI(true);

  if (dealerScore > 21) endRound('dealer-bust');
  else if (playerScore > dealerScore) endRound('win');
  else if (playerScore < dealerScore) endRound('lose');
  else endRound('push');
}

function endRound(outcome) {
  roundActive = false;
  updateUI(true);
  setPhase('end');

  if (outcome === 'blackjack') {
    payBlackjack();
  } else if (outcome === 'win') {
    payWin('YOU WIN!');
  } else if (outcome === 'dealer-bust') {
    payWin('DEALER BUSTS!');
  } else if (outcome === 'push') {
    returnBet();
  } else if (outcome === 'bust') {
    recordLoss('BUST!');
  } else {
    recordLoss('DEALER WINS');
  }

  finishBet();
}

function newRound() {
  if (balance === 0) {
    msg('No balance left! Reset to play.');
    return;
  }

  pendingBet = 0;
  playerHand = [];
  dealerHand = [];
  roundActive = false;

  clearResult();
  msg('');
  updateUI(false);
  setPhase('bet');
}

function resetGame() {
  balance = 500;
  bet = 0;
  pendingBet = 0;
  wins = 0;
  losses = 0;
  roundActive = false;
  playerHand = [];
  dealerHand = [];
  resetDeck();

  clearResult();
  msg('Game reset! Good luck.');
  updateUI(false);
  setPhase('bet');
}

// UI rendering
function isRed(suit) {
  return suit === '&hearts;' || suit === '&diams;';
}

function renderCard(card, hidden = false) {
  if (hidden) return '<div class="card hidden"></div>';

  return `<div class="card ${isRed(card.s) ? 'red' : 'black'}"><div class="card-rank">${card.r}</div><div class="card-suit">${card.s}</div></div>`;
}

function updateUI(revealDealer = false) {
  document.getElementById('bal').textContent = balance;
  document.getElementById('cur-bet').textContent = bet;
  document.getElementById('wins').textContent = wins;
  document.getElementById('losses').textContent = losses;
  document.getElementById('pending-bet').textContent = pendingBet;
  document.getElementById('dealer-cards').innerHTML = dealerHand.map((card, index) => renderCard(card, !revealDealer && index === 1)).join('');
  document.getElementById('player-cards').innerHTML = playerHand.map(card => renderCard(card)).join('');
  document.getElementById('player-score').textContent = calculateHandValue(playerHand) || '';
  document.getElementById('dealer-score').textContent = revealDealer ? calculateHandValue(dealerHand) : '?';

  updateAIAdvice();
}

function updateAIAdvice() {
  const panel = document.getElementById('ai-panel');
  const resultsBody = document.getElementById('ai-results');
  const simCount = document.getElementById('ai-sim-count');
  const recommendation = document.getElementById('ai-recommendation');
  const explanation = document.getElementById('ai-explanation');

  if (!panel || !resultsBody || !simCount || !recommendation || !explanation) {
    return;
  }

  if (!roundActive || isBust(playerHand)) {
    clearAIAdvice();
    return;
  }

  const odds = getActionOdds(getCurrentGameState());
  panel.classList.add('visible');
  simCount.textContent = `${DEFAULT_SIMULATION_COUNT.toLocaleString()} simulations`;
  resultsBody.innerHTML = odds.actionResults.map(result => renderAIAdviceRow(result, odds.recommendedAction)).join('');
  recommendation.textContent = odds.recommendedAction ? `Recommended action: ${formatActionName(odds.recommendedAction)}` : '';
  explanation.textContent = getAIExplanationText(odds);
}

function clearAIAdvice() {
  const panel = document.getElementById('ai-panel');
  const resultsBody = document.getElementById('ai-results');
  const simCount = document.getElementById('ai-sim-count');
  const recommendation = document.getElementById('ai-recommendation');
  const explanation = document.getElementById('ai-explanation');

  if (panel) panel.classList.remove('visible');
  if (resultsBody) resultsBody.innerHTML = '';
  if (simCount) simCount.textContent = '';
  if (recommendation) recommendation.textContent = '';
  if (explanation) explanation.textContent = '';
}

function renderAIAdviceRow(result, recommendedAction) {
  const recommendedClass = result.action === recommendedAction ? ' class="recommended"' : '';

  return `
    <tr${recommendedClass}>
      <td>${formatActionName(result.action)}</td>
      <td>${formatPercent(result.winRate)}</td>
      <td>${formatPercent(result.pushRate)}</td>
      <td>${formatPercent(result.lossRate)}</td>
      <td>${formatExpectedValue(result.expectedValue)}</td>
    </tr>
  `;
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatExpectedValue(value) {
  return value.toFixed(3);
}

function getAIExplanationText(odds) {
  if (!odds.recommendedAction) {
    return 'No recommendation is available for the current state.';
  }

  const bestResult = odds.actionResults.find(result => result.action === odds.recommendedAction);
  const bestEV = bestResult ? formatExpectedValue(bestResult.expectedValue) : '0.000';

  return `${odds.reason} EV is win rate minus loss rate, doubled for double down. This live panel uses Monte Carlo simulation; the supervised ML model is trained offline to approximate these recommendations. Current mode uses the full dealer hand for simulation. Best EV: ${bestEV}.`;
}

function setPhase(phase) {
  document.getElementById('bet-phase').style.display = phase === 'bet' ? '' : 'none';
  document.getElementById('play-phase').style.display = phase === 'play' ? '' : 'none';
  document.getElementById('end-phase').style.display = phase === 'end' ? '' : 'none';
}

function showResult(text, type) {
  document.getElementById('result-area').innerHTML = `<div class="result-banner ${type}">${text}</div>`;
}

function clearResult() {
  document.getElementById('result-area').innerHTML = '';
}

function msg(text) {
  document.getElementById('msg').textContent = text;
}

// Button and startup handlers
resetDeck();
updateUI(false);
setPhase('bet');
