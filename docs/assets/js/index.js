/**
 * Main JS file for Casper behaviours
 */

/*globals jQuery, document */
(function ($) {
    "use strict";

    $(document).ready(function(){

        $(".post-content").fitVids();
        
        // Calculates Reading Time (strip scripts, styles, and embedded chart data)
        var $cleanContent = $('.post-content').clone();
        $cleanContent.find('script, style, svg, noscript').remove();
        var cleanText = $cleanContent.text();
        var wordCount = cleanText.split(/\s+/).filter(function(w) { return w.length > 0; }).length;
        var readingTime = Math.max(1, Math.round(wordCount / 270));
        $('.post-reading-time').text(readingTime + ' min');
        
        // Creates Captions from Alt tags
        $(".post-content img").each(function() {
            // Let's put a caption if there is one
            if($(this).attr("alt") && !$(this).hasClass("emoji"))
              $(this).wrap('<figure class="image"></figure>')
              .after('<figcaption>'+$(this).attr("alt")+'</figcaption>');
        });
        
    });

}(jQuery));
