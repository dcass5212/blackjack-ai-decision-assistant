// RL policy integration — loads rl_policy_lookup.js (must precede this script)
// Hooks into the existing updateAIAdvice() without touching blackjack.js.

(function () {
  // -----------------------------------------------------------------------
  // Helpers that mirror the Python env logic
  // -----------------------------------------------------------------------

  function rlIsSoftHand(hand) {
    const hasAce = hand.some(c => c.r === 'A');
    if (!hasAce) return false;
    let hard = 0;
    for (const c of hand) {
      hard += c.r === 'A' ? 1 : getSimulationCardValue(c.r);
    }
    return hard + 10 <= 21;
  }

  function getRLAction(playerHand, dealerUpCard, canDoubleDown) {
    if (!playerHand || !dealerUpCard || typeof RL_POLICY === 'undefined') return null;
    const total = calculateSimulationHandValue(playerHand);
    if (total > 21) return null;
    const isSoft = rlIsSoftHand(playerHand) ? 1 : 0;
    const dealerVal = Math.min(getSimulationCardValue(dealerUpCard.r), 10);
    const canDouble = canDoubleDown ? 1 : 0;
    return RL_POLICY[`${total}-${isSoft}-${dealerVal}-${canDouble}`] || 'stand';
  }

  // -----------------------------------------------------------------------
  // Patch updateAIAdvice to append the RL verdict after the MC panel renders
  // -----------------------------------------------------------------------

  const _origUpdateAIAdvice = updateAIAdvice;

  window.updateAIAdvice = function () {
    _origUpdateAIAdvice();

    const verdict = document.getElementById('rl-verdict');
    if (!verdict) return;

    // Mirror the same guard conditions as the original function
    if (!roundActive || isSimulationBust(playerHand)) {
      verdict.classList.remove('visible');
      return;
    }

    const gameState = getCurrentGameState();
    const rlAction = getRLAction(gameState.playerHand, gameState.dealerUpCard, gameState.canDoubleDown);
    if (!rlAction) {
      verdict.classList.remove('visible');
      return;
    }

    // Read MC recommendation from the DOM (already set by _origUpdateAIAdvice)
    const recEl = document.getElementById('ai-recommendation');
    const recText = recEl ? recEl.textContent : '';
    let mcAction = null;
    if (recText.includes('Double')) mcAction = 'doubleDown';
    else if (recText.includes('Stand')) mcAction = 'stand';
    else if (recText.includes('Hit')) mcAction = 'hit';

    const labels = { hit: 'Hit', stand: 'Stand', doubleDown: 'Double Down' };
    document.getElementById('rl-action-text').textContent = labels[rlAction] || rlAction;

    const badge = document.getElementById('rl-verdict-badge');
    const agrees = mcAction && rlAction === mcAction;
    badge.textContent = agrees ? '✓ agrees with MC' : '⚡ differs from MC';
    badge.className = 'rl-verdict-badge ' + (agrees ? 'agree' : 'differ');

    verdict.classList.add('visible');
  };
})();
