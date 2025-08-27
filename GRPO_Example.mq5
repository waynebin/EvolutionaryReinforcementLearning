//+------------------------------------------------------------------+
//|                                                    GRPO_Example.mq5 |
//|   Demonstration of Group Relative Policy Optimization in MQL5     |
//|   Combining rules-based vs. model-based reward computations        |
//+------------------------------------------------------------------+
#property strict
#include <Trade/Trade.mqh>

//--------------------------------------------------------------------
// 1. Inputs / Parameters
//--------------------------------------------------------------------
input int     InpMaxBars         = 1000;     // Max bars to simulate in backtest
input int     InpGroupSize       = 8;        // Group size G for GRPO
input double  InpClipEpsilon     = 0.2;      // Clip ratio range in [1-ε, 1+ε]
input double  InpBetaKL          = 0.01;     // KL penalty coefficient
input double  InpLearnRate       = 0.001;    // LR for policy updates
input double  InpFormatWeight    = 0.5;      // Example weighting for "format" in reward
input bool    InpUseReferencePol = false;    // If true, keep a reference policy for KL

//--------------------------------------------------------------------
// 2. Policy + Old Policy + (Optional) Reference Policy
//--------------------------------------------------------------------
// Suppose we have 3 discrete actions: 0=Hold, 1=Buy, 2=Sell
// We'll have 3 continuous outputs for (SL, TP, LotSize) as an example.

#define DISCRETE_ACTIONS  3  // [Hold, Buy, Sell]
#define CONTINUOUS_PARAMS 3  // [StopLossOffset, TakeProfitOffset, LotSize]

// Keep them small for the example
double policyWeights[10];       // current policy
double oldPolicyWeights[10];    // snapshot
double refPolicyWeights[10];    // reference policy (optional for KL)
int    policyDim = 10;

//--------------------------------------------------------------------
// Forward declarations
//--------------------------------------------------------------------
double MathClamp(double val, double minv, double maxv);
int    SampleDiscreteAction(const double &pDiscrete[]);
void   PolicyForward(const double &weights[], const double &state[], double &outDiscrete[], double &outCont[]);

double RulesBasedDiscreteReward(int discreteAction);
double EvaluateContinuousParams(double slOff, double tpOff, double lotSize);
double ComputeReward(int discreteAction, double slOff, double tpOff, double lotSize);

double compute_reward(string response, string ground_truth);

//--------------------------------------------------------------------
// 3. Reward function demonstration
//--------------------------------------------------------------------
// Rules-based reward for discrete action
double RulesBasedDiscreteReward(int discreteAction)
{
   switch(discreteAction)
   {
      case 1: return  2.0; // buy
      case 2: return  2.0; // sell
      default: return 0.5; // hold
   }
}

// Model-based scoring for continuous parameters
double EvaluateContinuousParams(double slOff, double tpOff, double lotSize)
{
   double penaltySL  = MathMax(0.0, (slOff - 150.0)*0.01);  // if SL offset > 150 => penalty
   double penaltyTP  = MathMax(0.0, (tpOff - 300.0)*0.005); // if TP offset > 300 => penalty
   double penaltyLot = (lotSize > 2.0) ? (lotSize - 2.0)*2.0 : 0.0;

   double baseVal = 5.0; // baseline
   double finalVal = baseVal - penaltySL - penaltyTP - penaltyLot;

   return finalVal; // model-based reward
}

// Combined discrete + continuous reward
double ComputeReward(int discreteAction, double slOff, double tpOff, double lotSize)
{
   double discretePart   = RulesBasedDiscreteReward(discreteAction);
   double continuousPart = EvaluateContinuousParams(slOff, tpOff, lotSize);
   return (discretePart + continuousPart);
}

// Example placeholder showing how you might combine other reward terms
double compute_reward(string response, string ground_truth)
{
   double accuracy_reward = 0.0; // evaluate_correctness(response, ground_truth);
   double format_reward   = 0.0; // check_formatting(response);
   return accuracy_reward + format_reward * InpFormatWeight;
}

//--------------------------------------------------------------------
// 4. Math helpers and Policy Forward / Probability Computation
//--------------------------------------------------------------------

double MathClamp(double val, double minv, double maxv)
{
   if(val<minv) return minv;
   if(val>maxv) return maxv;
   return val;
}

// Given weights and state, outputs:
//  - probability distribution over 3 discrete actions
//  - predicted continuous parameters [SL, TP, LotSize]
void PolicyForward(const double &weights[], const double &state[], double &outDiscrete[], double &outCont[])
{
   ArrayResize(outDiscrete, DISCRETE_ACTIONS);
   ArrayInitialize(outDiscrete, 0.0);
   ArrayResize(outCont, CONTINUOUS_PARAMS);
   ArrayInitialize(outCont, 0.0);

   // Simple single-dimensional state
   double sVal = state[0];

   // Discrete part (linear + bias)
   for(int i=0; i<DISCRETE_ACTIONS; i++)
   {
      outDiscrete[i] = weights[i] * sVal + weights[6];
   }

   // Softmax
   double maxV = outDiscrete[0];
   for(int i=1; i<DISCRETE_ACTIONS; i++)
      if(outDiscrete[i]>maxV) maxV = outDiscrete[i];

   double sumExp = 0.0;
   for(int i=0; i<DISCRETE_ACTIONS; i++)
   {
      outDiscrete[i] = MathExp(outDiscrete[i] - maxV);
      sumExp += outDiscrete[i];
   }
   for(int i=0; i<DISCRETE_ACTIONS; i++)
      outDiscrete[i] /= MathMax(sumExp, 1e-12);

   // Continuous part (SL, TP, lotSize) - linear + bias
   outCont[0] = weights[3]*sVal + weights[7];  // SL offset
   outCont[1] = weights[4]*sVal + weights[8];  // TP offset
   outCont[2] = weights[5]*sVal + weights[9];  // Lot size

   // Clamp to ranges
   outCont[0] = MathClamp(outCont[0], 10.0, 300.0);
   outCont[1] = MathClamp(outCont[1], 20.0, 500.0);
   outCont[2] = MathClamp(outCont[2], 0.01, 5.0);
}

//--------------------------------------------------------------------
// 5. Sampling from old policy (Group Size G) & Reward Computation
//--------------------------------------------------------------------
struct GRPOBatch
{
   double states[];          // 1D: state value (e.g., close price)
   int    discreteActions[]; 
   double contSL[];
   double contTP[];
   double contLot[];
   double rewards[];
};

GRPOBatch g_batch;  // current group

int SampleDiscreteAction(const double &pDiscrete[])
{
   double rnd = (double)MathRand() / 32767.0;
   double cumsum = 0.0;
   for(int i=0; i<DISCRETE_ACTIONS; i++)
   {
      cumsum += pDiscrete[i];
      if(rnd <= cumsum)
         return i;
   }
   return DISCRETE_ACTIONS-1; // fallback
}

// Step function that samples G actions from old policy, evaluates reward, stores in g_batch
void CollectGroupSamples(const double &oldWeights[], int groupSize, int startShift)
{
   ArrayResize(g_batch.states,          groupSize);
   ArrayResize(g_batch.discreteActions, groupSize);
   ArrayResize(g_batch.contSL,          groupSize);
   ArrayResize(g_batch.contTP,          groupSize);
   ArrayResize(g_batch.contLot,         groupSize);
   ArrayResize(g_batch.rewards,         groupSize);

   for(int i=0; i<groupSize; i++)
   {
      int  shift = startShift + i;
      double st[1];
      st[0] = iClose(_Symbol, PERIOD_CURRENT, shift);

      double pDiscrete[], cParams[];
      PolicyForward(oldWeights, st, pDiscrete, cParams);

      int chosen = SampleDiscreteAction(pDiscrete);
      double rew = ComputeReward(chosen, cParams[0], cParams[1], cParams[2]);

      g_batch.states[i]          = st[0];
      g_batch.discreteActions[i] = chosen;
      g_batch.contSL[i]          = cParams[0];
      g_batch.contTP[i]          = cParams[1];
      g_batch.contLot[i]         = cParams[2];
      g_batch.rewards[i]         = rew;
   }
}

//--------------------------------------------------------------------
// 6. Compute GRPO Update
//--------------------------------------------------------------------
void GRPO_Update(double epsilon, double betaKL, double learnRate, bool useRefPol)
{
   int groupSize = ArraySize(g_batch.rewards);
   if(groupSize<=0) return;

   // 1) Mean & std of rewards
   double sumR=0.0;
   for(int i=0; i<groupSize; i++)
      sumR += g_batch.rewards[i];
   double meanR = sumR / MathMax(groupSize, 1);

   double sumSq=0.0;
   for(int i=0; i<groupSize; i++)
   {
      double diff = g_batch.rewards[i] - meanR;
      sumSq += diff*diff;
   }
   double variance = (groupSize>1) ? (sumSq/(groupSize-1)) : 1.0;
   double stdR     = MathSqrt(MathMax(variance, 1e-8));

   // Save a copy of current policy weights for isolation (not strictly needed here)
   double currentWeightsCopy[10];
   ArrayCopy(currentWeightsCopy, policyWeights, 0, 0, policyDim);

   for(int iter=0; iter<1; iter++)
   {
      double dTheta[10];
      ArrayInitialize(dTheta, 0.0);

      // KL against reference policy (very rough)
      double klTotal = 0.0;
      if(useRefPol)
      {
         for(int i=0; i<groupSize; i++)
         {
            double stRef[1]; stRef[0] = g_batch.states[i];
            double pRefDiscrete[], pRefCont[];
            PolicyForward(refPolicyWeights, stRef, pRefDiscrete, pRefCont);
            double pNowDiscrete[], pNowCont[];
            PolicyForward(policyWeights, stRef, pNowDiscrete, pNowCont);
            int act = g_batch.discreteActions[i];
            double pAiRef = MathMax(pRefDiscrete[act], 1e-12);
            double pAiNow = MathMax(pNowDiscrete[act], 1e-12);
            double klTerm = pAiRef * (MathLog(pAiRef/pAiNow) - 1.0);
            klTotal += klTerm;
         }
      }

      for(int i=0; i<groupSize; i++)
      {
         double A_i = (g_batch.rewards[i] - meanR)/stdR;

         double st[1]; st[0] = g_batch.states[i];
         double pNowD[], pNowC[];
         double pOldD[], pOldC[];

         PolicyForward(policyWeights,    st, pNowD, pNowC);
         PolicyForward(oldPolicyWeights, st, pOldD, pOldC);

         int act = g_batch.discreteActions[i];
         double pAiNow = MathMax(pNowD[act], 1e-12);
         double pAiOld = MathMax(pOldD[act], 1e-12);

         double ratio = pAiNow / pAiOld;
         double clippedRatio = MathClamp(ratio, 1.0-epsilon, 1.0+epsilon);

         double objUnclipped = ratio*A_i;
         double objClipped   = clippedRatio*A_i;
         double originalObj  = MathMin(objUnclipped, objClipped);

         // Finite-difference gradient approximation
         for(int paramIdx=0; paramIdx<policyDim; paramIdx++)
         {
            double origVal = policyWeights[paramIdx];
            double step = 1e-4;
            policyWeights[paramIdx] = origVal + step;

            double pShiftD[], pShiftC[];
            PolicyForward(policyWeights, st, pShiftD, pShiftC);
            double pAiShift = MathMax(pShiftD[act], 1e-12);
            double ratioShift   = (pAiShift / pAiOld);
            double clippedShift = MathClamp(ratioShift, 1.0-epsilon, 1.0+epsilon);
            double objShift     = MathMin(ratioShift*A_i, clippedShift*A_i);

            double gradParam = (objShift - originalObj)/step;
            dTheta[paramIdx] += gradParam;

            policyWeights[paramIdx] = origVal; // revert
         }
      }

      if(useRefPol)
      {
         for(int paramIdx=0; paramIdx<policyDim; paramIdx++)
         {
            dTheta[paramIdx] -= (betaKL * klTotal * 1e-4);
         }
      }

      for(int paramIdx=0; paramIdx<policyDim; paramIdx++)
      {
         policyWeights[paramIdx] += learnRate * dTheta[paramIdx];
      }
   }
}

//--------------------------------------------------------------------
// 7. OnInit / OnTick / OnDeinit
//--------------------------------------------------------------------
CTrade trade;
static int g_totalUpdates = 0;
static int g_startShift   = 0; // walk back through history bars for sampling

int OnInit()
{
   MathSrand((uint)GetTickCount());

   for(int i=0; i<policyDim; i++)
   {
      policyWeights[i]    = (MathRand()/32767.0 - 0.5)*0.01;
      oldPolicyWeights[i] = policyWeights[i];
      refPolicyWeights[i] = policyWeights[i];
   }

   Print("GRPO Example Initialized");
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   Print("GRPO Example Deinitialized. Updates performed: ", g_totalUpdates);
}

void OnTick()
{
   // Ensure enough bars exist for sampling
   int bars = Bars(_Symbol, PERIOD_CURRENT);
   if(bars <= InpGroupSize + g_startShift)
      return;

   // Stop if exceeded max bars window
   if(g_startShift >= InpMaxBars)
      return;

   // Collect a group from the old policy
   CollectGroupSamples(oldPolicyWeights, InpGroupSize, g_startShift);

   // Perform one GRPO update step
   GRPO_Update(InpClipEpsilon, InpBetaKL, InpLearnRate, InpUseReferencePol);

   // Update snapshot of old policy
   ArrayCopy(oldPolicyWeights, policyWeights, 0, 0, policyDim);

   // Optionally refresh reference policy slowly
   if(InpUseReferencePol)
      ArrayCopy(refPolicyWeights, policyWeights, 0, 0, policyDim);

   g_totalUpdates++;
   g_startShift += InpGroupSize; // advance window

   if((g_totalUpdates % 10)==0)
      PrintFormat("GRPO update %d done. Last state=%.5f, last reward=%.4f", g_totalUpdates, g_batch.states[InpGroupSize-1], g_batch.rewards[InpGroupSize-1]);
}

//+------------------------------------------------------------------+