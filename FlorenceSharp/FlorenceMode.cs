namespace FlorenceSharp
{
    public enum FlorenceMode
    {
        Caption,
        DetailedCaption,
        MoreDetailedCaption,
        OCR,
        OCRWithRegion,
        ObjectDetection,
        DenseRegionCaption,
        RegionProposal,
        
        // With inputs
        CaptionToPhraseGrounding,
        ReferringExpressionSegmentation,
        RegionToSegmentation,
        OpenVocabularyDetection,
        RegionToCategory,
        RegionToDescription,
        RegionToOCR,
    }
}