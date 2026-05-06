const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const projectRoot = path.resolve(__dirname, '..');

function readProjectFile(relativePath) {
  return fs.readFileSync(path.join(projectRoot, relativePath), 'utf8');
}

function loadAiContext() {
  const context = { console, Math };
  vm.createContext(context);
  vm.runInContext(readProjectFile('blackjack-ai.js'), context, { filename: 'blackjack-ai.js' });
  return context;
}

function createElement(id) {
  return {
    id,
    textContent: '',
    innerHTML: '',
    disabled: false,
    style: {},
    classList: {
      classes: new Set(),
      add(name) {
        this.classes.add(name);
      },
      remove(name) {
        this.classes.delete(name);
      },
      contains(name) {
        return this.classes.has(name);
      },
    },
  };
}

function loadGameContext() {
  const elements = new Map();
  const document = {
    getElementById(id) {
      if (!elements.has(id)) {
        elements.set(id, createElement(id));
      }

      return elements.get(id);
    },
  };
  const context = { console, Math, document };

  vm.createContext(context);
  vm.runInContext(readProjectFile('blackjack-ai.js'), context, { filename: 'blackjack-ai.js' });
  vm.runInContext(readProjectFile('blackjack.js'), context, { filename: 'blackjack.js' });

  return { context, document };
}

function card(rank, suit = 'spades') {
  return { r: rank, s: suit };
}

function normalize(value) {
  return JSON.parse(JSON.stringify(value));
}

function extractPythonList(source, name) {
  const match = source.match(new RegExp(`${name}\\s*=\\s*\\[([\\s\\S]*?)\\]`));
  assert.ok(match, `Expected to find ${name}`);

  return match[1]
    .split('\n')
    .map(line => line.trim().replace(/,$/, ''))
    .filter(Boolean)
    .flatMap(value => {
      if (value === '*RANK_COUNT_COLUMNS') {
        return extractPythonList(source, 'RANK_COUNT_COLUMNS');
      }

      const quoted = value.match(/^["'](.+)["']$/);
      return quoted ? [quoted[1]] : [];
    });
}

test('scores hands with aces as soft or hard values', () => {
  const { context } = loadGameContext();

  assert.equal(context.calculateHandValue([card('A'), card('9')]), 20);
  assert.equal(context.calculateHandValue([card('A'), card('9'), card('A')]), 21);
  assert.equal(context.calculateHandValue([card('A'), card('9'), card('5')]), 15);
});

test('detects blackjack and bust states', () => {
  const { context } = loadGameContext();

  assert.equal(context.isBlackjack([card('A'), card('K')]), true);
  assert.equal(context.isBlackjack([card('A'), card('9'), card('A')]), false);
  assert.equal(context.isBust([card('10'), card('Q'), card('2')]), true);
  assert.equal(context.isBust([card('10'), card('A')]), false);
});

test('returns legal actions based on bust and double-down eligibility', () => {
  const context = loadAiContext();

  assert.deepEqual(normalize(context.getLegalActions({
    playerHand: [card('10'), card('Q'), card('2')],
    canDoubleDown: true,
  })), []);
  assert.deepEqual(normalize(context.getLegalActions({
    playerHand: [card('10'), card('6')],
    canDoubleDown: false,
  })), ['hit', 'stand']);
  assert.deepEqual(normalize(context.getLegalActions({
    playerHand: [card('5'), card('6')],
    canDoubleDown: true,
  })), ['hit', 'stand', 'doubleDown']);
});

test('dealer draws until reaching at least 17', () => {
  const context = loadAiContext();
  const dealerHand = [card('10'), card('6')];
  const deck = [card('5')];

  context.playSimulationDealerTurn(dealerHand, deck);

  assert.equal(dealerHand.length, 3);
  assert.equal(context.calculateSimulationHandValue(dealerHand), 21);
  assert.equal(deck.length, 0);
});

test('current game state exposes only dealer upcard and keeps hidden card unseen', () => {
  const { context } = loadGameContext();

  context.loadTestHand('hard16');
  const state = context.getCurrentGameState();

  assert.equal(state.dealerHand.length, 1);
  assert.deepEqual(normalize(state.dealerUpCard), { r: '10', s: '&clubs;' });
  assert.deepEqual(normalize(state.dealerHand[0]), normalize(state.dealerUpCard));
  assert.equal(state.remainingDeck.some(hiddenCard => (
    hiddenCard.r === '7' && hiddenCard.s === '&diams;'
  )), true);
  assert.equal(state.remainingDeck.some(upcard => (
    upcard.r === '10' && upcard.s === '&clubs;'
  )), false);
});

test('Monte Carlo odds have a stable result shape and do not mutate input state', () => {
  const context = loadAiContext();
  const state = {
    playerHand: [card('10'), card('6')],
    dealerUpCard: card('10'),
    dealerHand: [card('10')],
    remainingDeck: [
      card('7'), card('2'), card('3'), card('4'), card('5'), card('6'),
      card('8'), card('9'), card('A'), card('K'), card('Q'), card('J'),
    ],
    canDoubleDown: true,
  };
  const before = JSON.stringify(state);

  const odds = context.getActionOdds(state, 100);

  assert.equal(JSON.stringify(state), before);
  assert.equal(typeof odds.recommendedAction, 'string');
  assert.deepEqual(normalize(odds.actionResults.map(result => result.action)), ['hit', 'stand', 'doubleDown']);

  for (const result of odds.actionResults) {
    assert.equal(result.simulations, 100);
    assert.equal(Number.isFinite(result.winRate), true);
    assert.equal(Number.isFinite(result.lossRate), true);
    assert.equal(Number.isFinite(result.pushRate), true);
    assert.equal(Number.isFinite(result.expectedValue), true);
    assert.equal(result.winRate + result.lossRate + result.pushRate >= 0.99, true);
    assert.equal(result.winRate + result.lossRate + result.pushRate <= 1.01, true);
  }
});

test('ML feature columns stay consistent across generation, training, and committed data', () => {
  const generatedColumns = extractPythonList(readProjectFile('ml/generate_dataset.py'), 'FEATURE_COLUMNS');
  const trainingColumns = extractPythonList(readProjectFile('ml/train_model.py'), 'FEATURE_COLUMNS');
  const inferenceColumns = extractPythonList(readProjectFile('ml/predict_policy.py'), 'FEATURE_COLUMNS');
  const csvHeader = readProjectFile('ml/data/blackjack_states.csv').split(/\r?\n/, 1)[0].split(',');

  assert.deepEqual(trainingColumns, generatedColumns);
  assert.deepEqual(inferenceColumns, generatedColumns);
  assert.deepEqual(csvHeader, [...generatedColumns, 'recommended_action']);
});
