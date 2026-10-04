#pragma once
#include "ctr_provider.h"
#include "ctr_data.h"
#include "ctr_helpers.h"
#include "split.h"

#include <catboost/libs/helpers/exception.h>

#include <util/generic/hash.h>
#include <util/generic/utility.h>

#include <functional>


class TStaticCtrProvider: public ICtrProvider {
public:
    TStaticCtrProvider() = default;
    explicit TStaticCtrProvider(TCtrData& ctrData)
        : CtrData(ctrData)
    {}
    ~TStaticCtrProvider() override {}

    bool HasNeededCtrs(TConstArrayRef<TModelCtr> neededCtrs) const override;

    void CalcCtrs(
        const TConstArrayRef<TModelCtr> neededCtrs,
        const TConstArrayRef<ui8> binarizedFeatures, // vector of binarized float & one hot features
        const TConstArrayRef<ui32> hashedCatFeatures,
        size_t docCount,
        TArrayRef<float> result) override;

    void SetupBinFeatureIndexes(
        const TConstArrayRef<TFloatFeature> floatFeatures,
        const TConstArrayRef<TOneHotFeature> oheFeatures,
        const TConstArrayRef<TCatFeature> catFeatures) override;
    bool IsSerializable() const override {
        return true;
    }

    void AddCtrCalcerData(TCtrValueTable&& valueTable) override {
        GetCacheForThisProvider().Invalidate();
        auto ctrBase = valueTable.ModelCtrBase;
        CtrData.LearnCtrs[ctrBase] = std::move(valueTable);
    }

    void DropUnusedTables(TConstArrayRef<TModelCtrBase> usedModelCtrBase) override {
        TCtrData ctrData;
        for (auto& base: usedModelCtrBase) {
            ctrData.LearnCtrs[base] = std::move(CtrData.LearnCtrs[base]);
        }
        DoSwap(CtrData, ctrData);
    }

    void Save(IOutputStream* out) const override {
        ::Save(out, CtrData);
    }

    void Load(IInputStream* inp) override {
        GetCacheForThisProvider().Invalidate();
        ::Load(inp, CtrData);
    }

    void LoadNonOwning(TMemoryInput* in) {
        GetCacheForThisProvider().Invalidate();
        CtrData.LoadNonOwning(in);
    }

    static TString ModelPartId() {
        return "static_provider_v1";
    }

    // Diagnostics for the per-thread cache registry: how many provider entries
    // the CALLING thread's registry currently holds. Diagnostic only, not used on
    // the hot path. Used to observe registry growth and address reuse on
    // long-lived threads that load and drop models repeatedly.
    static size_t GetCachedProviderCountForDiagnostics();
    static void GetGroupingCacheStatsForDiagnostics(size_t* hits, size_t* misses);

    TString ModelPartIdentifier() const override {
        return ModelPartId();
    }

    const THashMap<TFloatSplit, TBinFeatureIndexValue>& GetFloatFeatureIndexes() const {
        return FloatFeatureIndexes;
    }

    const THashMap<TOneHotSplit, TBinFeatureIndexValue>& GetOneHotFeatureIndexes() const {
        return OneHotFeatureIndexes;
    }

    virtual TIntrusivePtr<ICtrProvider> Clone() const override;

public:
    TCtrData CtrData;
private:
    /* Cache of the CompressModelCtrs grouping.
     *
     * CompressModelCtrs is called once per 128-document block and rebuilds the
     * same grouping of CTRs by projection every time, although it depends only
     * on the model. The grouping is a pure function of the Projection sequence
     * of the input array, so it can be reused when the same array comes back
     * unchanged.
     *
     * Correctness rests on two properties, both enforced here.
     *
     * The stored structures hold pointers INTO the caller-provided array, so
     * reuse needs a proof that the array still carries the same projection
     * sequence. That proof cannot come from the stored pointers: they are
     * &array[i], so comparing them with the array compares each element with
     * itself and accepts any in-place change of the grouping. ProjectionSnapshot
     * is therefore an independent copy of the sequence, taken when the grouping
     * is built, and reuse requires it to match element by element. An exact
     * element-wise comparison is used rather than a hash, so there is no
     * collision to reason about.
     *
     * The cache is per thread and keyed by provider. A cache in a provider
     * field was measured by ThreadSanitizer to be a data race when a provider
     * is shared by parallel prediction requests; a per-thread cache introduces
     * no shared mutable state and needs no synchronization.
     */
    struct TCtrGroupingCache {
        const TModelCtr* Source = nullptr;
        size_t SourceSize = 0;
        TVector<NCB::TCompressedModelCtr> Compressed;
        // Projection of every element as it was when Compressed was built.
        // Independent of the pointers inside Compressed, which refer to Source.
        TVector<TFeatureCombination> ProjectionSnapshot;

        void Invalidate() {
            Source = nullptr;
            SourceSize = 0;
            Compressed.clear();
            ProjectionSnapshot.clear();
        }
    };

    static bool IsCacheValid(const TCtrGroupingCache& cache,
                             const TConstArrayRef<TModelCtr>& neededCtrs);
    TCtrGroupingCache& GetCacheForThisProvider();
    const TVector<NCB::TCompressedModelCtr>& GetCachedGrouping(
        const TConstArrayRef<TModelCtr>& neededCtrs);

    THashMap<TFloatSplit, TBinFeatureIndexValue> FloatFeatureIndexes;
    THashMap<int, int> CatFeatureIndex;
    THashMap<TOneHotSplit, TBinFeatureIndexValue> OneHotFeatureIndexes;
};

class TStaticCtrOnFlightSerializationProvider: public ICtrProvider {
public:
    using TCtrParallelGenerator = std::function<void(const TVector<TModelCtrBase>&, TCtrDataStreamWriter*)>;

public:
    TStaticCtrOnFlightSerializationProvider(
        TVector<TModelCtrBase> ctrBases,
        TCtrParallelGenerator ctrParallelGenerator
    )
        : CtrBases(ctrBases)
        , CtrParallelGenerator(ctrParallelGenerator)
    {
    }
    ~TStaticCtrOnFlightSerializationProvider() = default;

    bool HasNeededCtrs(const TConstArrayRef<TModelCtr>) const override {
        return false;
    }

    void CalcCtrs(
        const TConstArrayRef<TModelCtr>,
        const TConstArrayRef<ui8>,
        const TConstArrayRef<ui32>,
        size_t,
        TArrayRef<float>) override {

        ythrow TCatBoostException()
            << "TStaticCtrOnFlightSerializationProvider is for streamed serialization only";
    }

    void SetupBinFeatureIndexes(
        const TConstArrayRef<TFloatFeature>,
        const TConstArrayRef<TOneHotFeature>,
        const TConstArrayRef<TCatFeature>) override {

        ythrow TCatBoostException()
            << "TStaticCtrOnFlightSerializationProvider is for streamed serialization only";
    }
    bool IsSerializable() const override {
        return true;
    }
    void AddCtrCalcerData(TCtrValueTable&& ) override {
        ythrow TCatBoostException()
            << "TStaticCtrOnFlightSerializationProvider is for streamed serialization only";
    }

    void DropUnusedTables(TConstArrayRef<TModelCtrBase>) override {
        ythrow TCatBoostException()
            << "TStaticCtrOnFlightSerializationProvider is for streamed serialization only";
    }

    void Save(IOutputStream* out) const override {
        TCtrDataStreamWriter streamWriter(out, CtrBases.size());
        CtrParallelGenerator(CtrBases, &streamWriter);
    }

    void Load(IInputStream*) override {
        ythrow TCatBoostException()
            << "TStaticCtrOnFlightSerializationProvider is for streamed serialization only";
    }

    TString ModelPartIdentifier() const override {
        return "static_provider_v1";
    }

private:
    TVector<TModelCtrBase> CtrBases;
    TCtrParallelGenerator CtrParallelGenerator;
};

TIntrusivePtr<TStaticCtrProvider> MergeStaticCtrProvidersData(
    const TVector<const TStaticCtrProvider*>& providers,
    ECtrTableMergePolicy mergePolicy);
