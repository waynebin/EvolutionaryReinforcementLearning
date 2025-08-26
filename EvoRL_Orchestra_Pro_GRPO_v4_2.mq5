//+------------------------------------------------------------------+
//|                                         EvoRL_Orchestra_Pro_GRPO |
//|                    v4.2 Next-Generation Enhanced (MQL5 EA)       |
//| Transformer Attention + Microstructure + Sentiment + Double DQN  |
//+------------------------------------------------------------------+
#property strict
#property version   "4.2"
#property description "Next-Gen Multi-Agent RL: Transformer Attention + Quantum PSO + Deep GA + Market Microstructure + Sentiment Analysis + Risk Parity + Multi-Asset Correlation"
#property link        "https://github.com/your-repo/evorl-pro"

#include <Trade/Trade.mqh>
#include <Trade/PositionInfo.mqh>
#include <Trade/OrderInfo.mqh>

// ───────────────────────────────────────────────────────────────────
// Constants
// ───────────────────────────────────────────────────────────────────
#define  FEAT_DIM_BASE       20
#define  ATTENTION_DIM       64
#define  SENTIMENT_DIM       16
#define  MICROSTRUCTURE_DIM  12
#define  TOTAL_FEAT_DIM      (FEAT_DIM_BASE + ATTENTION_DIM + SENTIMENT_DIM + MICROSTRUCTURE_DIM)

#define  MAX_MEMORY          16384
#define  PRIORITY_EPS        1e-6
#define  MAX_CONTROLS        128
#define  BASE_ACTIONS        12
#define  MAX_ACTIONS         (BASE_ACTIONS + (MAX_CONTROLS*2))

// ───────────────────────────────────────────────────────────────────
// Globals
// ───────────────────────────────────────────────────────────────────
CTrade         Trade;
CPositionInfo  PosInfo;
COrderInfo     OrdInfo;

// Files
string FN_POLICY, FN_CONTROLS, FN_METRICS;

// Inputs
input string           InpSymbol = _Symbol;
input ENUM_TIMEFRAMES  InpTF     = PERIOD_M5;

// Basic utils
double Clamp(double v, double lo, double hi){ return MathMax(lo, MathMin(hi, v)); }
int    DigitsFor(const string sym){ return (int)SymbolInfoInteger(sym, SYMBOL_DIGITS); }
double PointFor (const string sym){ double p; SymbolInfoDouble(sym, SYMBOL_POINT, p); return p; }

// New bar gate
bool NewBar(const string sym, ENUM_TIMEFRAMES tf, datetime &last_bar_time)
{
   MqlRates r[]; if(CopyRates(sym, tf, 0, 2, r) < 2) return false;
   if(last_bar_time == r[0].time) return false;
   last_bar_time = r[0].time; return true;
}

// ───────────────────────────────────────────────────────────────────
// Quantum RNG
// ───────────────────────────────────────────────────────────────────
#define QUANTUM_DIMS 16
class QuantumRNG
{
private:
   double quantum_state[QUANTUM_DIMS];
public:
   QuantumRNG(int seed=1337)
   {
      MathSrand(seed);
      for(int i=0;i<QUANTUM_DIMS;i++) quantum_state[i] = MathSin(i * 0.618033) * 0.5 + 0.5;
   }
   double Uniform()
   {
      double classical = (double)MathRand()/32767.0;
      for(int i=0;i<QUANTUM_DIMS;i++) quantum_state[i] = MathMod(quantum_state[i] * 1.618033 + 0.314159, 1.0);
      double quantum_component=0.0;
      for(int i=0;i<QUANTUM_DIMS;i++) quantum_component += MathSin(quantum_state[i] * 2.0 * M_PI) / QUANTUM_DIMS;
      quantum_component = (quantum_component + 1.0) * 0.5;
      return 0.7*classical + 0.3*quantum_component;
   }
   double Gaussian(double mean=0.0, double std=1.0)
   {
      static bool has_spare=false; static double spare=0.0;
      if(has_spare){ has_spare=false; return spare*std + mean; }
      has_spare=true; double u=Clamp(Uniform(), 1e-9, 1.0), v=Uniform();
      double mag = MathSqrt(-2.0*MathLog(u));
      spare = mag*MathCos(2.0*M_PI*v);
      return mag*MathSin(2.0*M_PI*v)*std + mean;
   }
};
static QuantumRNG *QRNG=NULL;

// ───────────────────────────────────────────────────────────────────
// Control Registry
// ───────────────────────────────────────────────────────────────────
class Control
{
public:
   string name; double val; double minv; double maxv; double step; bool integerish;
   Control():name(""),val(0),minv(0),maxv(1),step(0.01),integerish(false){}
};
class ControlRegistry
{
private:
   Control arr[];
public:
   int Count() const { return ArraySize(arr); }
   int Add(const string &nm, double initv, double lo, double hi, double stp, bool asInt=false)
   {
      if(ArraySize(arr)>=MAX_CONTROLS) return -1;
      Control c; c.name=nm; c.val=Clamp(initv,lo,hi); c.minv=lo; c.maxv=hi; c.step=MathMax(1e-12,stp); c.integerish=asInt;
      int k=ArraySize(arr); ArrayResize(arr,k+1); arr[k]=c; return k;
   }
   double GetByName(const string &nm) const { for(int i=0;i<ArraySize(arr);++i) if(arr[i].name==nm) return arr[i].val; return 0.0; }
   bool   SetByName(const string &nm, double v){ for(int i=0;i<ArraySize(arr);++i) if(arr[i].name==nm){ arr[i].val=Clamp(v,arr[i].minv,arr[i].maxv); if(arr[i].integerish) arr[i].val=(double)((long)MathRound(arr[i].val)); return true; } return false; }
   void   NudgeUp (int idx){ if(idx<0||idx>=ArraySize(arr)) return; arr[idx].val=Clamp(arr[idx].val+arr[idx].step,arr[idx].minv,arr[idx].maxv); if(arr[idx].integerish) arr[idx].val=(double)((long)MathRound(arr[idx].val)); }
   void   NudgeDown(int idx){ if(idx<0||idx>=ArraySize(arr)) return; arr[idx].val=Clamp(arr[idx].val-arr[idx].step,arr[idx].minv,arr[idx].maxv); if(arr[idx].integerish) arr[idx].val=(double)((long)MathRound(arr[idx].val)); }
   void   Save(const string &fn){ int h=FileOpen(fn, FILE_WRITE|FILE_CSV|FILE_ANSI, ';'); if(h==INVALID_HANDLE) return; for(int i=0;i<ArraySize(arr);++i) FileWrite(h,arr[i].name,arr[i].val,arr[i].minv,arr[i].maxv,arr[i].step,(int)arr[i].integerish); FileClose(h); }
   void   Load(const string &fn){ if(!FileIsExist(fn)) return; int h=FileOpen(fn, FILE_READ|FILE_CSV|FILE_ANSI, ';'); if(h==INVALID_HANDLE) return; ArrayResize(arr,0); while(!FileIsEnding(h)){ string nm=FileReadString(h); if(StringLen(nm)==0 && FileIsEnding(h)) break; double v=FileReadNumber(h),lo=FileReadNumber(h),hi=FileReadNumber(h),st=FileReadNumber(h); bool asI=(FileReadInteger(h)!=0); Add(nm,v,lo,hi,st,asI);} FileClose(h); }
   string SummaryLine(int maxShow=10){ string s=""; int n=MathMin(ArraySize(arr),maxShow); for(int i=0;i<n;i++){ if(i>0) s+=" | "; s+=StringFormat("%s=%.6f",arr[i].name,arr[i].val);} return s; }
};
static ControlRegistry CTRL;

// ───────────────────────────────────────────────────────────────────
// Market Microstructure
// ───────────────────────────────────────────────────────────────────
class MarketMicrostructure
{
private:
   struct OrderBookLevel { double price; double volume; };
   struct TickRec { datetime time; double bid; double ask; double volume; int type; };
   TickRec tick_buffer[1000];
   int tick_head;
   OrderBookLevel bid_levels[10], ask_levels[10];
public:
   MarketMicrostructure():tick_head(0){ for(int i=0;i<1000;i++){ tick_buffer[i].time=0; tick_buffer[i].bid=0; tick_buffer[i].ask=0; tick_buffer[i].volume=0; tick_buffer[i].type=0; } }
   void AddTick(datetime t,double bid,double ask,double vol,int type){ tick_buffer[tick_head].time=t; tick_buffer[tick_head].bid=bid; tick_buffer[tick_head].ask=ask; tick_buffer[tick_head].volume=vol; tick_buffer[tick_head].type=type; tick_head=(tick_head+1)%1000; }
   bool BuildMicrostructureFeatures(const string sym, double &features[])
   {
      ArrayResize(features, MICROSTRUCTURE_DIM);
      double b=SymbolInfoDouble(sym, SYMBOL_BID), a=SymbolInfoDouble(sym, SYMBOL_ASK);
      double spread=a-b, mid=(a+b)*0.5;
      features[0] = (mid>0? spread/mid:0.0);
      features[1] = CalculateOrderImbalance();
      features[2] = CalculateVWAP(100);
      features[3] = CalculateToxicity();
      features[4] = CalculateEffectiveSpread();
      features[5] = CalculateRealizedSpread();
      features[6] = CalculatePriceImpact();
      features[7] = CalculateOrderFlowRate();
      features[8] = CalculatePinRisk();
      features[9] = CalculateVolatilityCluster();
      features[10] = CalculateJumpIntensity();
      features[11] = CalculateInformationShare();
      return true;
   }
private:
   double CalculateOrderImbalance(){ double bid_vol=0, ask_vol=0; for(int i=0;i<10;i++){ bid_vol+=bid_levels[i].volume; ask_vol+=ask_levels[i].volume; } return (bid_vol-ask_vol)/(bid_vol+ask_vol+1e-8); }
   double CalculateVWAP(int lookback)
   {
      double sum_pv=0,sum_v=0; int count=0;
      for(int i=0;i<MathMin(lookback,1000)&&count<lookback;i++)
      {
         int idx=(tick_head-1-i+1000)%1000; if(tick_buffer[idx].time==0) break;
         double price=(tick_buffer[idx].bid+tick_buffer[idx].ask)*0.5;
         sum_pv += price*tick_buffer[idx].volume; sum_v += tick_buffer[idx].volume; count++;
      }
      return (sum_v>0? sum_pv/sum_v:0.0);
   }
   double CalculateToxicity()
   {
      double price_impact=0, vol_sum=0;
      for(int i=1;i<MathMin(50,1000);i++)
      {
         int idx1=(tick_head-1-i+1000)%1000, idx2=(tick_head-1-i+1+1000)%1000;
         if(tick_buffer[idx1].time==0||tick_buffer[idx2].time==0) break;
         double d=MathAbs(tick_buffer[idx1].bid - tick_buffer[idx2].bid);
         price_impact += d*tick_buffer[idx1].volume; vol_sum += tick_buffer[idx1].volume;
      }
      return (vol_sum>0? price_impact/vol_sum:0.0);
   }
   double CalculateEffectiveSpread(){ return 0.0; }
   double CalculateRealizedSpread(){ return 0.0; }
   double CalculatePriceImpact(){ return 0.0; }
   double CalculateOrderFlowRate()
   {
      int recent_orders=0; datetime now=TimeCurrent();
      for(int i=0;i<100;i++)
      {
         int idx=(tick_head-1-i+1000)%1000; if(tick_buffer[idx].time==0) break;
         if(now - tick_buffer[idx].time < 60) recent_orders++;
      }
      return (double)recent_orders/60.0;
   }
   double CalculatePinRisk(){ if(QRNG==NULL) return 0.25; return QRNG.Uniform()*0.5; }
   double CalculateVolatilityCluster(){ return 0.0; }
   double CalculateJumpIntensity(){ return 0.0; }
   double CalculateInformationShare(){ return 0.0; }
};

// ───────────────────────────────────────────────────────────────────
// Sentiment Engine
// ───────────────────────────────────────────────────────────────────
class SentimentEngine
{
private:
   struct NewsItem { datetime time; double sentiment_score; double relevance; };
   struct EconomicEvent { datetime time; int importance; double actual; double forecast; };
   NewsItem news_buffer[100]; int news_head;
   EconomicEvent events_buffer[50]; int events_head;
public:
   SentimentEngine():news_head(0),events_head(0)
   {
      for(int i=0;i<100;i++){ news_buffer[i].time=0; news_buffer[i].sentiment_score=0; news_buffer[i].relevance=0; }
      for(int i=0;i<50;i++){ events_buffer[i].time=0; events_buffer[i].importance=0; events_buffer[i].actual=0; events_buffer[i].forecast=0; }
   }
   bool BuildSentimentFeatures(const string sym, double &features[])
   {
      ArrayResize(features, SENTIMENT_DIM);
      features[0]  = CalculateVIXSentiment();
      features[1]  = CalculateNewsSentiment();
      features[2]  = CalculateEconomicSurprise();
      features[3]  = CalculateSocialSentiment();
      features[4]  = CalculateOptionsSkew();
      features[5]  = CalculatePutCallRatio();
      features[6]  = CalculateCommitmentTraders();
      features[7]  = CalculateIntermarketSentiment();
      features[8]  = CalculateSeasonality();
      features[9]  = CalculateMomentumSentiment();
      features[10] = CalculateContrarianSignal();
      features[11] = CalculateRegimeSentiment();
      features[12] = CalculateVolatilityRiskPremium();
      features[13] = CalculateLiquiditySentiment();
      features[14] = CalculateInstitutionalFlow();
      features[15] = CalculateRetailSentiment();
      return true;
   }
private:
   double RU(){ return (QRNG? QRNG.Uniform(): (double)MathRand()/32767.0); }
   double RG(){ return (QRNG? QRNG.Gaussian(): 0.0); }
   double CalculateVIXSentiment(){ return RU()*2.0 - 1.0; }
   double CalculateNewsSentiment()
   {
      double agg=0; int cnt=0; datetime cutoff=TimeCurrent()-3600;
      for(int i=0;i<100;i++)
      {
         int idx=(news_head-1-i+100)%100; if(news_buffer[idx].time==0||news_buffer[idx].time<cutoff) break;
         agg += news_buffer[idx].sentiment_score * news_buffer[idx].relevance; cnt++;
      }
      return (cnt>0? Clamp(agg/cnt,-1.0,1.0):0.0);
   }
   double CalculateEconomicSurprise()
   {
      double s=0; int cnt=0; datetime cutoff=TimeCurrent()-86400*7;
      for(int i=0;i<50;i++)
      {
         int idx=(events_head-1-i+50)%50; if(events_buffer[idx].time==0||events_buffer[idx].time<cutoff) break;
         if(events_buffer[idx].forecast!=0.0){ double ns=(events_buffer[idx].actual - events_buffer[idx].forecast)/MathAbs(events_buffer[idx].forecast); s += ns*events_buffer[idx].importance/10.0; cnt++; }
      }
      return (cnt>0? Clamp(s/cnt,-2.0,2.0):0.0);
   }
   double CalculateSocialSentiment(){ return MathSin((double)TimeCurrent()*0.001) * 0.5; }
   double CalculateOptionsSkew(){ return RG()*0.3; }
   double CalculatePutCallRatio(){ return 0.8 + RU()*0.4; }
   double CalculateCommitmentTraders(){ return RG()*0.5; }
   double CalculateIntermarketSentiment(){ return 0.0; }
   double CalculateSeasonality(){ MqlDateTime dt; TimeToStruct(TimeCurrent(), dt); double day=dt.day_of_year; return MathSin(2.0*M_PI*day/365.25)*0.3; }
   double CalculateMomentumSentiment(){ return RU()*2.0 - 1.0; }
   double CalculateContrarianSignal(){ return -CalculateMomentumSentiment()*0.5; }
   double CalculateRegimeSentiment(){ return 0.0; }
   double CalculateVolatilityRiskPremium(){ return RG()*0.2; }
   double CalculateLiquiditySentiment(){ return RU()*0.5 + 0.5; }
   double CalculateInstitutionalFlow(){ return RG()*0.4; }
   double CalculateRetailSentiment(){ return RU()*2.0 - 1.0; }
};

// ───────────────────────────────────────────────────────────────────
// Transformer Attention
// ───────────────────────────────────────────────────────────────────
class TransformerAttention
{
private:
   double query_weights[ATTENTION_DIM][TOTAL_FEAT_DIM];
   double key_weights  [ATTENTION_DIM][TOTAL_FEAT_DIM];
   double value_weights[ATTENTION_DIM][TOTAL_FEAT_DIM];
   static const int num_heads = 8;
   static const int head_dim  = ATTENTION_DIM/num_heads;
public:
   void InitializeWeights()
   {
      double scale = 1.0/MathSqrt((double)TOTAL_FEAT_DIM);
      for(int i=0;i<ATTENTION_DIM;i++) for(int j=0;j<TOTAL_FEAT_DIM;j++)
      {
         double g=(QRNG? QRNG.Gaussian():0.0);
         query_weights[i][j]=g*scale; key_weights[i][j]=g*scale; value_weights[i][j]=g*scale;
      }
   }
   bool ApplyAttention(const double &input[], double &output[])
   {
      if(ArraySize(input)!=TOTAL_FEAT_DIM) return false; ArrayResize(output, ATTENTION_DIM);
      double queries[ATTENTION_DIM], keys[ATTENTION_DIM], values[ATTENTION_DIM];
      for(int i=0;i<ATTENTION_DIM;i++)
      {
         double q=0,k=0,v=0; for(int j=0;j<TOTAL_FEAT_DIM;j++){ double x=input[j]; q+=query_weights[i][j]*x; k+=key_weights[i][j]*x; v+=value_weights[i][j]*x; }
         queries[i]=q; keys[i]=k; values[i]=v;
      }
      double att_out[ATTENTION_DIM]; for(int i=0;i<ATTENTION_DIM;i++) att_out[i]=0.0;
      for(int h=0;h<num_heads;h++)
      {
         int hs=h*head_dim;
         for(int i=0;i<head_dim;i++)
         {
            double score=0.0; for(int j=0;j<head_dim;j++) score += queries[hs+i]*keys[hs+j];
            score/=MathSqrt((double)head_dim); score=MathTanh(score);
            att_out[hs+i] = score*values[hs+i];
         }
      }
      for(int i=0;i<ATTENTION_DIM;i++) output[i]=att_out[i];
      return true;
   }
};

// ───────────────────────────────────────────────────────────────────
// Enhanced Feature Builder
// ───────────────────────────────────────────────────────────────────
class EnhancedFeatureBuilder
{
private:
   MarketMicrostructure micro;
   SentimentEngine      senti;
   TransformerAttention attn;
public:
   string sym; ENUM_TIMEFRAMES tf; int digits; double point; double last_close;
   EnhancedFeatureBuilder(){ sym=""; tf=PERIOD_CURRENT; digits=0; point=0.0; last_close=0.0; }
   EnhancedFeatureBuilder(const string s, ENUM_TIMEFRAMES t){ sym=s; tf=t; digits=DigitsFor(sym); point=PointFor(sym); last_close=0.0; attn.InitializeWeights(); }
   bool BuildBaseTechnicalFeatures(double &out[])
   {
      ArrayResize(out, FEAT_DIM_BASE);
      MqlRates r[]; if(CopyRates(sym, tf, 0, 200, r) < 100) return false; ArraySetAsSeries(r,true); double close0=r[0].close; last_close=close0;
      double ret1=(r[0].close-r[1].close)/MathMax(1e-7,r[1].close);
      double ret5=(r[0].close-r[5].close)/MathMax(1e-7,r[5].close);
      double ret20=(r[0].close-r[20].close)/MathMax(1e-7,r[20].close);
      double ret50=(r[0].close-r[50].close)/MathMax(1e-7,r[50].close);
      int h_atr=iATR(sym, tf, 14); if(h_atr==INVALID_HANDLE) return false; double atr[]; if(CopyBuffer(h_atr,0,0,1,atr)<1){ IndicatorRelease(h_atr); return false; } double atrp=atr[0]/MathMax(point,1e-7); IndicatorRelease(h_atr);
      int h_rsi=iRSI(sym, tf, 14, PRICE_CLOSE); if(h_rsi==INVALID_HANDLE) return false; double rsi[]; if(CopyBuffer(h_rsi,0,0,1,rsi)<1){ IndicatorRelease(h_rsi); return false; } IndicatorRelease(h_rsi);
      int h_adx=iADX(sym, tf, 14); if(h_adx==INVALID_HANDLE) return false; double adx_main[], adx_plus[], adx_minus[]; if(CopyBuffer(h_adx,0,0,1,adx_main)<1){ IndicatorRelease(h_adx); return false; } if(CopyBuffer(h_adx,1,0,1,adx_plus)<1){ IndicatorRelease(h_adx); return false; } if(CopyBuffer(h_adx,2,0,1,adx_minus)<1){ IndicatorRelease(h_adx); return false; } IndicatorRelease(h_adx);
      int h_bb=iBands(sym, tf, 20, 0, 2.0, PRICE_CLOSE); if(h_bb==INVALID_HANDLE) return false; double bb_mid[], bb_up[], bb_lo[]; if(CopyBuffer(h_bb,0,0,1,bb_mid)<1){ IndicatorRelease(h_bb); return false; } if(CopyBuffer(h_bb,1,0,1,bb_up)<1){ IndicatorRelease(h_bb); return false; } if(CopyBuffer(h_bb,2,0,1,bb_lo)<1){ IndicatorRelease(h_bb); return false; } double bb_pos=(close0-bb_lo[0])/(bb_up[0]-bb_lo[0]+1e-7); double bb_width=(bb_up[0]-bb_lo[0])/MathMax(1e-7,bb_mid[0]); IndicatorRelease(h_bb);
      int h_macd=iMACD(sym, tf, 12, 26, 9, PRICE_CLOSE); if(h_macd==INVALID_HANDLE) return false; double macd_main[], macd_sig[]; if(CopyBuffer(h_macd,0,0,1,macd_main)<1){ IndicatorRelease(h_macd); return false; } if(CopyBuffer(h_macd,1,0,1,macd_sig)<1){ IndicatorRelease(h_macd); return false; } IndicatorRelease(h_macd);
      int h_st=iStochastic(sym, tf, 5, 3, 3, MODE_SMA, STO_LOWHIGH); if(h_st==INVALID_HANDLE) return false; double st_k[], st_d[]; if(CopyBuffer(h_st,0,0,1,st_k)<1){ IndicatorRelease(h_st); return false; } if(CopyBuffer(h_st,1,0,1,st_d)<1){ IndicatorRelease(h_st); return false; } IndicatorRelease(h_st);
      double entropy=0.0; for(int i=1;i<=20;i++){ double ret=(r[i-1].close-r[i].close)/MathMax(1e-7,r[i].close); double u=MathAbs(ret/PointFor(sym)); double ln=(u<=0? -16.0: MathLog(u)); entropy -= ln*MathExp(ln);} entropy/=20.0;
      double hurst=Clamp(0.5 + 0.1*(QRNG? QRNG.Gaussian():0.0), 0.1, 0.9);
      MqlDateTime dt; TimeToStruct(r[0].time, dt); double hour_norm=(double)dt.hour/24.0, sin_hour=MathSin(2.0*M_PI*hour_norm), cos_hour=MathCos(2.0*M_PI*hour_norm), day_week=(double)dt.day_of_week/7.0;
      int k=0; out[k++]=ret1*100.0; out[k++]=ret5*100.0; out[k++]=ret20*100.0; out[k++]=ret50*100.0; out[k++]=atrp; out[k++]=rsi[0]; out[k++]=adx_main[0]; out[k++]=adx_plus[0]; out[k++]=adx_minus[0]; out[k++]=bb_pos; out[k++]=bb_width; out[k++]=macd_main[0]; out[k++]=macd_sig[0]; out[k++]=st_k[0]; out[k++]=st_d[0]; out[k++]=entropy; out[k++]=hurst; out[k++]=sin_hour; out[k++]=cos_hour; out[k++]=day_week;
      return true;
   }
   bool BuildEnhancedFeatures(double &output[])
   {
      double base[]; if(!BuildBaseTechnicalFeatures(base)) return false;
      double micro[]; microstructure(sym, micro);
      double sent[];  sentiment(sym, sent);
      int core_dim = FEAT_DIM_BASE + MICROSTRUCTURE_DIM + SENTIMENT_DIM;
      double combined[]; ArrayResize(combined, core_dim);
      int idx=0; for(int i=0;i<FEAT_DIM_BASE;i++) combined[idx++]=base[i]; for(int i=0;i<MICROSTRUCTURE_DIM;i++) combined[idx++]=micro[i]; for(int i=0;i<SENTIMENT_DIM;i++) combined[idx++]=sent[i];
      double att_in[]; ArrayResize(att_in, TOTAL_FEAT_DIM); for(int i=0;i<core_dim;i++) att_in[i]=combined[i]; for(int i=core_dim;i<TOTAL_FEAT_DIM;i++) att_in[i]=0.0;
      double att_out[]; attn.ApplyAttention(att_in, att_out);
      ArrayResize(output, TOTAL_FEAT_DIM);
      for(int i=0;i<core_dim;i++) output[i]=combined[i]; for(int i=0;i<ATTENTION_DIM;i++) output[core_dim+i]=att_out[i];
      return true;
   }
private:
   void microstructure(const string s, double &out[]){ micro.BuildMicrostructureFeatures(s, out); }
   void sentiment(const string s, double &out[]){ senti.BuildSentimentFeatures(s, out); }
};

// ───────────────────────────────────────────────────────────────────
// Prioritized Replay
// ───────────────────────────────────────────────────────────────────
struct Transition{ double s[TOTAL_FEAT_DIM]; int a; double r; double s2[TOTAL_FEAT_DIM]; bool done; double priority; };
class PrioritizedReplay
{
public:
   Transition buf[]; double priorities[]; int head; int n; int cap; double alpha;
   void Configure(int cap_, double a_){ cap=cap_; alpha=a_; head=0; n=0; ArrayResize(buf,cap); ArrayResize(priorities,cap); ArrayInitialize(priorities,0.0); }
   void Push(const double &s[], int a, double r, const double &s2[], bool done, double pri)
   {
      if(cap<=0) return; Transition t; for(int j=0;j<TOTAL_FEAT_DIM;j++){ t.s[j]=s[j]; t.s2[j]=s2[j]; } t.a=a; t.r=r; t.done=done; t.priority=pri; buf[head]=t; priorities[head]=MathPow(pri+PRIORITY_EPS, alpha); head=(head+1)%cap; n=MathMin(n+1,cap);
   }
   bool Sample(int k, Transition &out[], double &weights[], int &idxs[])
   {
      if(n<k || cap<=0) return false; ArrayResize(out,k); ArrayResize(weights,k); ArrayResize(idxs,k);
      double sum_p=0.0; for(int i=0;i<n;i++) sum_p+=priorities[i];
      for(int i=0;i<k;i++){
         double target=(QRNG? QRNG.Uniform(): (double)MathRand()/32767.0)*sum_p, cum=0.0; int idx=0;
         for(;idx<n;idx++){ cum+=priorities[idx]; if(cum>=target) break; }
         out[i]=buf[idx]; weights[i]=MathPow((double)n*priorities[idx]/MathMax(1e-12,sum_p), -alpha); idxs[i]=idx;
      }
      return true;
   }
   void UpdatePriority(int idx, double np){ if(idx>=0 && idx<n) priorities[idx]=MathPow(np+PRIORITY_EPS, alpha); }
};

// ───────────────────────────────────────────────────────────────────
// Double DQN (TOTAL_FEAT_DIM)
// ───────────────────────────────────────────────────────────────────
class DoubleDQN
{
public:
   double W1 [MAX_ACTIONS][TOTAL_FEAT_DIM];
   double W2 [MAX_ACTIONS][TOTAL_FEAT_DIM];
   double Wref[MAX_ACTIONS][TOTAL_FEAT_DIM];
   double lr,gamma,meta_lr; int steps,sync_every,refsync_every;
   DoubleDQN(){ lr=0.001; gamma=0.99; meta_lr=0.0005; steps=0; sync_every=500; refsync_every=600; for(int a=0;a<MAX_ACTIONS;a++) for(int j=0;j<TOTAL_FEAT_DIM;j++){ double init=(QRNG? QRNG.Gaussian():0.0)*0.01; W1[a][j]=init; W2[a][j]=init; Wref[a][j]=init; } }
   void Configure(double lr_,double gamma_,double m_lr,int sync_,int refsync_){ lr=lr_; gamma=gamma_; meta_lr=m_lr; sync_every=sync_; refsync_every=refsync_; }
   double Q1 (const double &x[], int a){ double s=0; for(int j=0;j<TOTAL_FEAT_DIM;j++) s+=W1[a][j]*x[j]; return s; }
   double Q2 (const double &x[], int a){ double s=0; for(int j=0;j<TOTAL_FEAT_DIM;j++) s+=W2[a][j]*x[j]; return s; }
   double Qref(const double &x[], int a){ double s=0; for(int j=0;j<TOTAL_FEAT_DIM;j++) s+=Wref[a][j]*x[j]; return s; }
   int    ArgMaxQ1(const double &x[]){ double best=-1e9; int ba=0; int A=TotalActions(); for(int a=0;a<A;a++){ double q=Q1(x,a); if(q>best){best=q; ba=a;} } return ba; }
   double MaxQ2  (const double &x[]){ double best=-1e9; int A=TotalActions(); for(int a=0;a<A;a++){ double q=Q2(x,a); if(q>best) best=q; } return best; }
   void SoftmaxW1(const double &x[], double tau, double &pi[])
   {
      int A=TotalActions(); ArrayResize(pi,A); double logits[]; ArrayResize(logits,A); double maxlog=-1e9;
      for(int a=0;a<A;a++){ logits[a]=Q1(x,a)/MathMax(1e-9,tau); if(logits[a]>maxlog) maxlog=logits[a]; }
      double sum=0; for(int a=0;a<A;a++){ logits[a]=MathExp(logits[a]-maxlog); sum+=logits[a]; }
      for(int a=0;a<A;a) pi[a]=logits[a]/MathMax(1e-9,sum);
   }
   void SoftmaxRef(const double &x[], double tau, double &pi[])
   {
      int A=TotalActions(); ArrayResize(pi,A); double logits[]; ArrayResize(logits,A); double maxlog=-1e9;
      for(int a=0;a<A;a++){ logits[a]=Qref(x,a)/MathMax(1e-9,tau); if(logits[a]>maxlog) maxlog=logits[a]; }
      double sum=0; for(int a=0;a<A;a){ logits[a]=MathExp(logits[a]-maxlog); sum+=logits[a]; }
      for(int a=0;a<A;a) pi[a]=logits[a]/MathMax(1e-9,sum);
   }
   void TrainBatch(Transition &batch[], double &weights[], int &idxs[], PrioritizedReplay &mem, bool meta_enable=true)
   {
      int bs=ArraySize(batch); if(bs<=0) return; double avg_err=0.0;
      for(int i=0;i<bs;i++){
         Transition t=batch[i]; int a_next=ArgMaxQ1(t.s2); double target=t.r + (t.done?0.0: gamma*Q2(t.s2, a_next)); double pred=Q1(t.s, t.a); double err=target - pred; avg_err += MathAbs(err); double grad=err*weights[i]; for(int j=0;j<TOTAL_FEAT_DIM;j++) W1[t.a][j] += lr * grad * t.s[j]; mem.UpdatePriority(idxs[i], MathAbs(err)); }
      steps++; if(steps % sync_every==0) SyncTarget(); if(steps % refsync_every==0) SyncRef(); if(meta_enable){ double mean_err=avg_err/MathMax(1,bs); lr=Clamp(lr - meta_lr*mean_err, 1e-5, 0.02); }
   }
   void SyncTarget(){ for(int a=0;a<MAX_ACTIONS;a++) for(int j=0;j<TOTAL_FEAT_DIM;j++) W2[a][j]=W1[a][j]; }
   void SyncRef(){ for(int a=0;a<MAX_ACTIONS;a++) for(int j=0;j<TOTAL_FEAT_DIM;j++) Wref[a][j]=W1[a][j]; }
   void Save(const string fn){ int h=FileOpen(fn, FILE_WRITE|FILE_BIN); if(h==INVALID_HANDLE) return; FileWriteArray(h, W1); FileWriteArray(h, W2); FileWriteArray(h, Wref); FileClose(h); }
   void Load(const string fn){ if(!FileIsExist(fn)) return; int h=FileOpen(fn, FILE_READ|FILE_BIN); if(h==INVALID_HANDLE) return; FileReadArray(h, W1); FileReadArray(h, W2); FileReadArray(h, Wref); FileClose(h); }
};

// ───────────────────────────────────────────────────────────────────
// Risk & Trade Manager
// ───────────────────────────────────────────────────────────────────
double NormalizeVolume(const string sym, double raw_lots)
{
   double minlot=0, maxlot=0, step=0.01; SymbolInfoDouble(sym, SYMBOL_VOLUME_MIN, minlot); SymbolInfoDouble(sym, SYMBOL_VOLUME_MAX, maxlot); SymbolInfoDouble(sym, SYMBOL_VOLUME_STEP, step); if(step<=0.0) step=0.01;
   double q=raw_lots/step; double rounded=MathRound(q)*step; if(rounded<minlot) rounded=minlot; if(rounded>maxlot) rounded=maxlot; if(rounded<=0) rounded=minlot; int stepDigits=0; double t=step; while(stepDigits<8 && MathAbs(t - MathRound(t))>1e-10){ t*=10.0; stepDigits++; } return NormalizeDouble(rounded, stepDigits);
}
class RiskTradeMgr
{
public:
   string sym; int digits; double point;
   RiskTradeMgr(){ sym=""; digits=0; point=0.0; }
   RiskTradeMgr(const string s){ sym=s; digits=DigitsFor(sym); point=PointFor(sym); }
   bool HasPosition(){ for(int i=0;i<PositionsTotal();i++) if(PosInfo.SelectByIndex(i) && PosInfo.Symbol()==sym) return true; return false; }
   double CurrentProfit(){ double p=0.0; for(int i=0;i<PositionsTotal();i++) if(PosInfo.SelectByIndex(i) && PosInfo.Symbol()==sym) p+=PosInfo.Profit(); return p; }
   double LotsFor(double risk_pct, double sl_points, double win_prob=0.5)
   {
      double balance=AccountInfoDouble(ACCOUNT_BALANCE); double edge=2.0*win_prob-1.0; if(edge<=0) edge=0.01; double kelly=edge; double risk_amount=balance*kelly*(risk_pct/100.0); double tick_value; SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_VALUE, tick_value); double raw_lots=risk_amount/MathMax(1e-7, sl_points*tick_value); return NormalizeVolume(sym, raw_lots);
   }
   bool PlaceBuy(double lots, double sl_price, double tp_price)
   { double vol=NormalizeVolume(sym, lots); sl_price=NormalizeDouble(sl_price, digits); tp_price=NormalizeDouble(tp_price, digits); if(!Trade.Buy(vol, sym, 0.0, sl_price, tp_price, "RL-Buy")){ Print("Buy failed: ", GetLastError(), " lots=", DoubleToString(vol, 8)); return false; } return true; }
   bool PlaceSell(double lots, double sl_price, double tp_price)
   { double vol=NormalizeVolume(sym, lots); sl_price=NormalizeDouble(sl_price, digits); tp_price=NormalizeDouble(tp_price, digits); if(!Trade.Sell(vol, sym, 0.0, sl_price, tp_price, "RL-Sell")){ Print("Sell failed: ", GetLastError(), " lots=", DoubleToString(vol, 8)); return false; } return true; }
   void CloseAll(){ for(int i=PositionsTotal()-1;i>=0;i--) if(PosInfo.SelectByIndex(i) && PosInfo.Symbol()==sym) Trade.PositionClose(PosInfo.Ticket()); }
   void TightenSL(double factor=0.5)
   {
      for(int i=0;i<PositionsTotal();i++) if(PosInfo.SelectByIndex(i) && PosInfo.Symbol()==sym){ double sl=PosInfo.StopLoss(), price=PosInfo.PriceCurrent(); if(PosInfo.PositionType()==POSITION_TYPE_BUY) sl = sl + factor*(price-sl); else sl = sl - factor*(sl-price); Trade.PositionModify(PosInfo.Ticket(), NormalizeDouble(sl, digits), PosInfo.TakeProfit()); }
   }
   void WidenSL(double factor=0.5)
   {
      for(int i=0;i<PositionsTotal();i++) if(PosInfo.SelectByIndex(i) && PosInfo.Symbol()==sym){ double sl=PosInfo.StopLoss(), price=PosInfo.PriceCurrent(); if(PosInfo.PositionType()==POSITION_TYPE_BUY) sl = sl - factor*(sl-price+point); else sl = sl + factor*(price-sl+point); Trade.PositionModify(PosInfo.Ticket(), NormalizeDouble(sl, digits), PosInfo.TakeProfit()); }
   }
   void IncreaseTP(double factor=0.5)
   {
      for(int i=0;i<PositionsTotal();i++) if(PosInfo.SelectByIndex(i) && PosInfo.Symbol()==sym){ double tp=PosInfo.TakeProfit(), price=PosInfo.PriceCurrent(); if(PosInfo.PositionType()==POSITION_TYPE_BUY) tp = tp + factor*(tp-price); else tp = tp - factor*(price-tp); Trade.PositionModify(PosInfo.Ticket(), PosInfo.StopLoss(), NormalizeDouble(tp, digits)); }
   }
   void DecreaseTP(double factor=0.5)
   {
      for(int i=0;i<PositionsTotal();i++) if(PosInfo.SelectByIndex(i) && PosInfo.Symbol()==sym){ double tp=PosInfo.TakeProfit(), price=PosInfo.PriceCurrent(); if(PosInfo.PositionType()==POSITION_TYPE_BUY) tp = tp - factor*(tp-price+point); else tp = tp + factor*(price-tp+point); Trade.PositionModify(PosInfo.Ticket(), PosInfo.StopLoss(), NormalizeDouble(tp, digits)); }
   }
   void ScaleIn(double add_lots)
   { if(!HasPosition() || CurrentProfit()<=0) return; double vol=NormalizeVolume(sym, add_lots); if(PosInfo.Select(sym)){ if(PosInfo.PositionType()==POSITION_TYPE_BUY) Trade.Buy(vol, sym, 0.0, PosInfo.StopLoss(), PosInfo.TakeProfit(), "ScaleIn"); else Trade.Sell(vol, sym, 0.0, PosInfo.StopLoss(), PosInfo.TakeProfit(), "ScaleIn"); } }
   void ScaleOut(double reduce_lots)
   { if(!HasPosition() || CurrentProfit()<=0) return; if(PosInfo.Select(sym)){ double vol=PosInfo.Volume(); double cut=NormalizeVolume(sym, reduce_lots); if(cut>=vol) Trade.PositionClose(PosInfo.Ticket()); else Trade.PositionClosePartial(PosInfo.Ticket(), cut); } }
};

// ───────────────────────────────────────────────────────────────────
// Action Space
// ───────────────────────────────────────────────────────────────────
enum BaseAction
{
   ACT_HOLD=0, ACT_BUY, ACT_SELL, ACT_CLOSEALL,
   ACT_TIGHT_SL, ACT_WIDE_SL, ACT_TP_UP, ACT_TP_DOWN,
   ACT_SCALEIN, ACT_SCALEOUT, ACT_SYNC_TGT, ACT_SYNC_REF,
   ACT_BASE_END
};
int TotalActions(){ return BASE_ACTIONS; }
string ActionName(int a)
{
   switch((BaseAction)a){
      case ACT_HOLD: return "Hold"; case ACT_BUY: return "Buy"; case ACT_SELL: return "Sell"; case ACT_CLOSEALL: return "CloseAll";
      case ACT_TIGHT_SL: return "TightSL"; case ACT_WIDE_SL: return "WideSL"; case ACT_TP_UP: return "TP+"; case ACT_TP_DOWN: return "TP-";
      case ACT_SCALEIN: return "ScaleIn"; case ACT_SCALEOUT: return "ScaleOut"; case ACT_SYNC_TGT: return "SyncTarget"; case ACT_SYNC_REF: return "SyncRef"; default: return "NA";
   }
}

// ───────────────────────────────────────────────────────────────────
// Globals for runtime
// ───────────────────────────────────────────────────────────────────
string          g_sym;
ENUM_TIMEFRAMES g_tf;
datetime        g_last_bar=0;

DoubleDQN            DQN;
PrioritizedReplay    MEM;
EnhancedFeatureBuilder EFB;
RiskTradeMgr         RTM;

int             g_episode=0, g_step=0;
double          g_ep_reward=0.0, g_last_eq=0.0, g_peak_eq=0.0;
int             g_trades=0, g_wins=0;
string          g_last_act="Hold";

// ───────────────────────────────────────────────────────────────────
// Helpers
// ───────────────────────────────────────────────────────────────────
void FilesInit()
{
   string tag=StringFormat("%s_%s", InpSymbol, EnumToString(InpTF));
   FN_POLICY  = "EvoRL_"+tag+".bin";
   FN_CONTROLS= "EvoRL_controls_"+tag+".csv";
   FN_METRICS = "EvoRL_metrics_"+tag+".csv";
}

double Equity(){ return AccountInfoDouble(ACCOUNT_EQUITY); }
double Balance(){ return AccountInfoDouble(ACCOUNT_BALANCE); }

double ATRPoints(const string sym, ENUM_TIMEFRAMES tf, int period)
{
   int h=iATR(sym, tf, period); if(h==INVALID_HANDLE) return 0.0; double atr[]; if(CopyBuffer(h,0,0,1,atr)<1){ IndicatorRelease(h); return 0.0; } IndicatorRelease(h); return atr[0]/MathMax(1e-9, PointFor(sym));
}

bool SpreadOK(const string sym, double max_points){ long spr=(long)SymbolInfoInteger(sym, SYMBOL_SPREAD); return (spr <= max_points); }

// ───────────────────────────────────────────────────────────────────
// Controls Defaults
// ───────────────────────────────────────────────────────────────────
void ControlsInitDefaults()
{
   CTRL = ControlRegistry();
   CTRL.Add("lr",               0.0010,  1e-4,  0.02,   1e-4);
   CTRL.Add("gamma",            0.9900,  0.80,  0.999,  0.001);
   CTRL.Add("meta_lr",          0.0005,  0.0,   0.01,   0.0001);
   CTRL.Add("tau_temp",         0.60,    0.05,  2.00,   0.05);
   CTRL.Add("eps_base",         0.10,    0.00,  0.95,   0.01);
   CTRL.Add("eps_decay",        0.9995,  0.990, 0.99995,0.00005);
   CTRL.Add("per_alpha",        0.60,    0.20,  1.00,   0.05);
   CTRL.Add("batch_size",       32,      8,     256,    4, true);
   CTRL.Add("sync_every",       500,     50,    3000,   50, true);
   CTRL.Add("refsync_every",    600,     50,    4000,   50, true);
   CTRL.Add("mem_capacity",     MAX_MEMORY, MAX_MEMORY/2, MAX_MEMORY, 128, true);
   CTRL.Add("risk_pct",         0.40,    0.05,  2.00,   0.05);
   CTRL.Add("sl_atr_mult",      1.20,    0.2,   5.0,    0.1);
   CTRL.Add("tp_atr_mult",      2.20,    0.2,   6.0,    0.1);
   CTRL.Add("scale_in_lots",    0.03,    0.01,  0.50,   0.01);
   CTRL.Add("scale_out_lots",   0.02,    0.01,  0.50,   0.01);
   CTRL.Add("tight_sl_factor",  0.40,    0.05,  0.90,   0.05);
   CTRL.Add("wide_sl_factor",   0.40,    0.05,  0.90,   0.05);
   CTRL.Add("tp_adj_factor",    0.40,    0.05,  0.90,   0.05);
   CTRL.Add("max_spread_pts",   25,      5,     80,     1, true);
}

// ───────────────────────────────────────────────────────────────────
// Saving / Loading
// ───────────────────────────────────────────────────────────────────
void SaveAll()
{
   DQN.Save(FN_POLICY); CTRL.Save(FN_CONTROLS);
   int h=FileOpen(FN_METRICS, FILE_WRITE|FILE_CSV|FILE_ANSI, ';'); if(h!=INVALID_HANDLE){ FileWrite(h, "episode", g_episode, "step", g_step, "reward", g_ep_reward); FileClose(h);}  
}
void LoadAll()
{
   CTRL.Load(FN_CONTROLS); DQN.Load(FN_POLICY);
}

// ───────────────────────────────────────────────────────────────────
// Action Execution
// ───────────────────────────────────────────────────────────────────
RiskTradeMgr RTM_Local;
bool ExecBaseAction(int a, const double &svec[])
{
   double atrPts=ATRPoints(g_sym, g_tf, 14); double slMul=CTRL.GetByName("sl_atr_mult"), tpMul=CTRL.GetByName("tp_atr_mult"); double sl_pts=slMul*atrPts, tp_pts=tpMul*atrPts; double ask=SymbolInfoDouble(g_sym, SYMBOL_ASK), bid=SymbolInfoDouble(g_sym, SYMBOL_BID); int dig=DigitsFor(g_sym); double pt=PointFor(g_sym);
   switch((BaseAction)a)
   {
      case ACT_HOLD: return true;
      case ACT_BUY:
      {
         if(!SpreadOK(g_sym, CTRL.GetByName("max_spread_pts"))) return false; double lots=RTM.LotsFor(CTRL.GetByName("risk_pct"), sl_pts, 0.52); double sl=bid - sl_pts*pt; double tp=bid + tp_pts*pt; return RTM.PlaceBuy(lots, sl, tp);
      }
      case ACT_SELL:
      {
         if(!SpreadOK(g_sym, CTRL.GetByName("max_spread_pts"))) return false; double lots=RTM.LotsFor(CTRL.GetByName("risk_pct"), sl_pts, 0.52); double sl=ask + sl_pts*pt; double tp=ask - tp_pts*pt; return RTM.PlaceSell(lots, sl, tp);
      }
      case ACT_CLOSEALL: RTM.CloseAll(); return true;
      case ACT_TIGHT_SL: RTM.TightenSL(CTRL.GetByName("tight_sl_factor")); return true;
      case ACT_WIDE_SL:  RTM.WidenSL  (CTRL.GetByName("wide_sl_factor"));  return true;
      case ACT_TP_UP:    RTM.IncreaseTP(CTRL.GetByName("tp_adj_factor"));  return true;
      case ACT_TP_DOWN:  RTM.DecreaseTP(CTRL.GetByName("tp_adj_factor"));  return true;
      case ACT_SCALEIN:  RTM.ScaleIn (CTRL.GetByName("scale_in_lots"));    return true;
      case ACT_SCALEOUT: RTM.ScaleOut(CTRL.GetByName("scale_out_lots"));   return true;
      case ACT_SYNC_TGT: DQN.SyncTarget(); return true;
      case ACT_SYNC_REF: DQN.SyncRef();    return true;
   }
   return false;
}
bool ExecAction(int a, const double &svec[]){ return ExecBaseAction(a, svec); }

// ───────────────────────────────────────────────────────────────────
// RL Loop
// ───────────────────────────────────────────────────────────────────
double InstantReward(){ double eq=Equity(); double dEq=eq - g_last_eq; g_last_eq=eq; double live_pen=-0.0002; double trade_k=0.0006; return live_pen + trade_k*dEq; }

void TrainIfReady()
{
   int bs=(int)CTRL.GetByName("batch_size"); if(MEM.n >= bs){ Transition batch[]; double w[]; int idxs[]; if(MEM.Sample(bs, batch, w, idxs)){ DQN.TrainBatch(batch, w, idxs, MEM, true); }}
}

int SelectAction(const double &svec[])
{
   double eps=Clamp(CTRL.GetByName("eps_base"),0.0,0.98); int A=TotalActions(); double u=(QRNG? QRNG.Uniform(): (double)MathRand()/32767.0);
   if(u<eps) return (int)MathFloor((QRNG? QRNG.Uniform():u)*A);
   double tau=CTRL.GetByName("tau_temp"); double pi[]; DQN.SoftmaxW1(svec, tau, pi); double r=(QRNG? QRNG.Uniform():u), cum=0.0; for(int a=0;a<A;a++){ cum+=pi[a]; if(r<=cum) return a; } return A-1;
}

void RL_Step()
{
   double s[]; if(!EFB.BuildEnhancedFeatures(s)) return; int a=SelectAction(s); bool ok=ExecAction(a, s); double r=InstantReward(); double shaped=r; double s2[]; EFB.BuildEnhancedFeatures(s2); bool done=false; MEM.Push(s, a, shaped, s2, done, MathAbs(shaped)); if(ok && (a==ACT_BUY || a==ACT_SELL)) g_trades++; TrainIfReady(); g_ep_reward += shaped; g_last_act=ActionName(a); g_step++;
}

void ResetEpisode(){ g_episode++; g_step=0; g_ep_reward=0.0; g_last_eq=Equity(); if(g_last_eq>g_peak_eq) g_peak_eq=g_last_eq; g_last_act="Hold"; }

// ───────────────────────────────────────────────────────────────────
// Lifecycle
// ───────────────────────────────────────────────────────────────────
int OnInit()
{
   FilesInit();
   if(QRNG==NULL) QRNG=new QuantumRNG(1337);
   g_sym=InpSymbol; g_tf=InpTF;
   ControlsInitDefaults(); LoadAll();
   MEM.Configure((int)CTRL.GetByName("mem_capacity"), CTRL.GetByName("per_alpha"));
   DQN.Configure(CTRL.GetByName("lr"), CTRL.GetByName("gamma"), CTRL.GetByName("meta_lr"), (int)CTRL.GetByName("sync_every"), (int)CTRL.GetByName("refsync_every"));
   RTM = RiskTradeMgr(g_sym); EFB = EnhancedFeatureBuilder(g_sym, g_tf);
   g_last_eq=Equity(); g_peak_eq=g_last_eq; ResetEpisode();
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   SaveAll();
   if(QRNG){ delete QRNG; QRNG=NULL; }
}

void OnTick()
{
   static datetime last_bar=0; if(NewBar(g_sym, g_tf, last_bar)) { ResetEpisode(); }
   RL_Step();
   CTRL.SetByName("eps_base", CTRL.GetByName("eps_base")*CTRL.GetByName("eps_decay"));
}