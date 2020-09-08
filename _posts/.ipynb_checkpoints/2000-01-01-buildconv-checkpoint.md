---
layout: post
title: Exploration in Modeling with SF 311
subtitle: Trying to predict 
cover-img: /assets/img/rainbow.png
tags: [SF311, opendata, public, bay area, quality of life, modeling,predictions]
---

<html>
<head><meta charset="utf-8" />

<title>buildconv</title>

<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/2.0.3/jquery.min.js"></script>

<script>
(function() {
  function addWidgetsRenderer() {
    var mimeElement = document.querySelector('script[type="application/vnd.jupyter.widget-view+json"]');
    var scriptElement = document.createElement('script');
    var widgetRendererSrc = 'https://unpkg.com/@jupyter-widgets/html-manager@*/dist/embed-amd.js';
    var widgetState;

    // Fallback for older version:
    try {
      widgetState = mimeElement && JSON.parse(mimeElement.innerHTML);

      if (widgetState && (widgetState.version_major < 2 || !widgetState.version_major)) {
        widgetRendererSrc = 'https://unpkg.com/jupyter-js-widgets@*/dist/embed.js';
      }
    } catch(e) {}

    scriptElement.src = widgetRendererSrc;
    document.body.appendChild(scriptElement);
  }

  document.addEventListener('DOMContentLoaded', addWidgetsRenderer);
}());
</script>

<style type="text/css">
    /*!
*
* Twitter Bootstrap
*
*/
/*!
 * Bootstrap v3.3.7 (http://getbootstrap.com)
 * Copyright 2011-2016 Twitter, Inc.
 * Licensed under MIT (https://github.com/twbs/bootstrap/blob/master/LICENSE)
 */
/*! normalize.css v3.0.3 | MIT License | github.com/necolas/normalize.css */
html {
  font-family: sans-serif;
  -ms-text-size-adjust: 100%;
  -webkit-text-size-adjust: 100%;
}
body {
  margin: 0;
}
article,
aside,
details,
figcaption,
figure,
footer,
header,
hgroup,
main,
menu,
nav,
section,
summary {
  display: block;
}
audio,
canvas,
progress,
video {
  display: inline-block;
  vertical-align: baseline;
}
audio:not([controls]) {
  display: none;
  height: 0;
}
[hidden],
template {
  display: none;
}
a {
  background-color: transparent;
}
a:active,
a:hover {
  outline: 0;
}
abbr[title] {
  border-bottom: 1px dotted;
}
b,
strong {
  font-weight: bold;
}
dfn {
  font-style: italic;
}
h1 {
  font-size: 2em;
  margin: 0.67em 0;
}
mark {
  background: #ff0;
  color: #000;
}
small {
  font-size: 80%;
}
sub,
sup {
  font-size: 75%;
  line-height: 0;
  position: relative;
  vertical-align: baseline;
}
sup {
  top: -0.5em;
}
sub {
  bottom: -0.25em;
}
img {
  border: 0;
}
svg:not(:root) {
  overflow: hidden;
}
figure {
  margin: 1em 40px;
}
hr {
  box-sizing: content-box;
  height: 0;
}
pre {
  overflow: auto;
}
code,
kbd,
pre,
samp {
  font-family: monospace, monospace;
  font-size: 1em;
}
button,
input,
optgroup,
select,
textarea {
  color: inherit;
  font: inherit;
  margin: 0;
}
button {
  overflow: visible;
}
button,
select {
  text-transform: none;
}
button,
html input[type="button"],
input[type="reset"],
input[type="submit"] {
  -webkit-appearance: button;
  cursor: pointer;
}
button[disabled],
html input[disabled] {
  cursor: default;
}
button::-moz-focus-inner,
input::-moz-focus-inner {
  border: 0;
  padding: 0;
}
input {
  line-height: normal;
}
input[type="checkbox"],
input[type="radio"] {
  box-sizing: border-box;
  padding: 0;
}
input[type="number"]::-webkit-inner-spin-button,
input[type="number"]::-webkit-outer-spin-button {
  height: auto;
}
input[type="search"] {
  -webkit-appearance: textfield;
  box-sizing: content-box;
}
input[type="search"]::-webkit-search-cancel-button,
input[type="search"]::-webkit-search-decoration {
  -webkit-appearance: none;
}
fieldset {
  border: 1px solid #c0c0c0;
  margin: 0 2px;
  padding: 0.35em 0.625em 0.75em;
}
legend {
  border: 0;
  padding: 0;
}
textarea {
  overflow: auto;
}
optgroup {
  font-weight: bold;
}
table {
  border-collapse: collapse;
  border-spacing: 0;
}
td,
th {
  padding: 0;
}
/*! Source: https://github.com/h5bp/html5-boilerplate/blob/master/src/css/main.css */
@media print {
  *,
  *:before,
  *:after {
    background: transparent !important;
    box-shadow: none !important;
    text-shadow: none !important;
  }
  a,
  a:visited {
    text-decoration: underline;
  }
  a[href]:after {
    content: " (" attr(href) ")";
  }
  abbr[title]:after {
    content: " (" attr(title) ")";
  }
  a[href^="#"]:after,
  a[href^="javascript:"]:after {
    content: "";
  }
  pre,
  blockquote {
    border: 1px solid #999;
    page-break-inside: avoid;
  }
  thead {
    display: table-header-group;
  }
  tr,
  img {
    page-break-inside: avoid;
  }
  img {
    max-width: 100% !important;
  }
  p,
  h2,
  h3 {
    orphans: 3;
    widows: 3;
  }
  h2,
  h3 {
    page-break-after: avoid;
  }
  .navbar {
    display: none;
  }
  .btn > .caret,
  .dropup > .btn > .caret {
    border-top-color: #000 !important;
  }
  .label {
    border: 1px solid #000;
  }
  .table {
    border-collapse: collapse !important;
  }
  .table td,
  .table th {
    background-color: #fff !important;
  }
  .table-bordered th,
  .table-bordered td {
    border: 1px solid #ddd !important;
  }
}
@font-face {
  font-family: 'Glyphicons Halflings';
  src: url('../components/bootstrap/fonts/glyphicons-halflings-regular.eot');
  src: url('../components/bootstrap/fonts/glyphicons-halflings-regular.eot?#iefix') format('embedded-opentype'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.woff2') format('woff2'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.woff') format('woff'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.ttf') format('truetype'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.svg#glyphicons_halflingsregular') format('svg');
}
.glyphicon {
  position: relative;
  top: 1px;
  display: inline-block;
  font-family: 'Glyphicons Halflings';
  font-style: normal;
  font-weight: normal;
  line-height: 1;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
.glyphicon-asterisk:before {
  content: "\002a";
}
.glyphicon-plus:before {
  content: "\002b";
}
.glyphicon-euro:before,
.glyphicon-eur:before {
  content: "\20ac";
}
.glyphicon-minus:before {
  content: "\2212";
}
.glyphicon-cloud:before {
  content: "\2601";
}
.glyphicon-envelope:before {
  content: "\2709";
}
.glyphicon-pencil:before {
  content: "\270f";
}
.glyphicon-glass:before {
  content: "\e001";
}
.glyphicon-music:before {
  content: "\e002";
}
.glyphicon-search:before {
  content: "\e003";
}
.glyphicon-heart:before {
  content: "\e005";
}
.glyphicon-star:before {
  content: "\e006";
}
.glyphicon-star-empty:before {
  content: "\e007";
}
.glyphicon-user:before {
  content: "\e008";
}
.glyphicon-film:before {
  content: "\e009";
}
.glyphicon-th-large:before {
  content: "\e010";
}
.glyphicon-th:before {
  content: "\e011";
}
.glyphicon-th-list:before {
  content: "\e012";
}
.glyphicon-ok:before {
  content: "\e013";
}
.glyphicon-remove:before {
  content: "\e014";
}
.glyphicon-zoom-in:before {
  content: "\e015";
}
.glyphicon-zoom-out:before {
  content: "\e016";
}
.glyphicon-off:before {
  content: "\e017";
}
.glyphicon-signal:before {
  content: "\e018";
}
.glyphicon-cog:before {
  content: "\e019";
}
.glyphicon-trash:before {
  content: "\e020";
}
.glyphicon-home:before {
  content: "\e021";
}
.glyphicon-file:before {
  content: "\e022";
}
.glyphicon-time:before {
  content: "\e023";
}
.glyphicon-road:before {
  content: "\e024";
}
.glyphicon-download-alt:before {
  content: "\e025";
}
.glyphicon-download:before {
  content: "\e026";
}
.glyphicon-upload:before {
  content: "\e027";
}
.glyphicon-inbox:before {
  content: "\e028";
}
.glyphicon-play-circle:before {
  content: "\e029";
}
.glyphicon-repeat:before {
  content: "\e030";
}
.glyphicon-refresh:before {
  content: "\e031";
}
.glyphicon-list-alt:before {
  content: "\e032";
}
.glyphicon-lock:before {
  content: "\e033";
}
.glyphicon-flag:before {
  content: "\e034";
}
.glyphicon-headphones:before {
  content: "\e035";
}
.glyphicon-volume-off:before {
  content: "\e036";
}
.glyphicon-volume-down:before {
  content: "\e037";
}
.glyphicon-volume-up:before {
  content: "\e038";
}
.glyphicon-qrcode:before {
  content: "\e039";
}
.glyphicon-barcode:before {
  content: "\e040";
}
.glyphicon-tag:before {
  content: "\e041";
}
.glyphicon-tags:before {
  content: "\e042";
}
.glyphicon-book:before {
  content: "\e043";
}
.glyphicon-bookmark:before {
  content: "\e044";
}
.glyphicon-print:before {
  content: "\e045";
}
.glyphicon-camera:before {
  content: "\e046";
}
.glyphicon-font:before {
  content: "\e047";
}
.glyphicon-bold:before {
  content: "\e048";
}
.glyphicon-italic:before {
  content: "\e049";
}
.glyphicon-text-height:before {
  content: "\e050";
}
.glyphicon-text-width:before {
  content: "\e051";
}
.glyphicon-align-left:before {
  content: "\e052";
}
.glyphicon-align-center:before {
  content: "\e053";
}
.glyphicon-align-right:before {
  content: "\e054";
}
.glyphicon-align-justify:before {
  content: "\e055";
}
.glyphicon-list:before {
  content: "\e056";
}
.glyphicon-indent-left:before {
  content: "\e057";
}
.glyphicon-indent-right:before {
  content: "\e058";
}
.glyphicon-facetime-video:before {
  content: "\e059";
}
.glyphicon-picture:before {
  content: "\e060";
}
.glyphicon-map-marker:before {
  content: "\e062";
}
.glyphicon-adjust:before {
  content: "\e063";
}
.glyphicon-tint:before {
  content: "\e064";
}
.glyphicon-edit:before {
  content: "\e065";
}
.glyphicon-share:before {
  content: "\e066";
}
.glyphicon-check:before {
  content: "\e067";
}
.glyphicon-move:before {
  content: "\e068";
}
.glyphicon-step-backward:before {
  content: "\e069";
}
.glyphicon-fast-backward:before {
  content: "\e070";
}
.glyphicon-backward:before {
  content: "\e071";
}
.glyphicon-play:before {
  content: "\e072";
}
.glyphicon-pause:before {
  content: "\e073";
}
.glyphicon-stop:before {
  content: "\e074";
}
.glyphicon-forward:before {
  content: "\e075";
}
.glyphicon-fast-forward:before {
  content: "\e076";
}
.glyphicon-step-forward:before {
  content: "\e077";
}
.glyphicon-eject:before {
  content: "\e078";
}
.glyphicon-chevron-left:before {
  content: "\e079";
}
.glyphicon-chevron-right:before {
  content: "\e080";
}
.glyphicon-plus-sign:before {
  content: "\e081";
}
.glyphicon-minus-sign:before {
  content: "\e082";
}
.glyphicon-remove-sign:before {
  content: "\e083";
}
.glyphicon-ok-sign:before {
  content: "\e084";
}
.glyphicon-question-sign:before {
  content: "\e085";
}
.glyphicon-info-sign:before {
  content: "\e086";
}
.glyphicon-screenshot:before {
  content: "\e087";
}
.glyphicon-remove-circle:before {
  content: "\e088";
}
.glyphicon-ok-circle:before {
  content: "\e089";
}
.glyphicon-ban-circle:before {
  content: "\e090";
}
.glyphicon-arrow-left:before {
  content: "\e091";
}
.glyphicon-arrow-right:before {
  content: "\e092";
}
.glyphicon-arrow-up:before {
  content: "\e093";
}
.glyphicon-arrow-down:before {
  content: "\e094";
}
.glyphicon-share-alt:before {
  content: "\e095";
}
.glyphicon-resize-full:before {
  content: "\e096";
}
.glyphicon-resize-small:before {
  content: "\e097";
}
.glyphicon-exclamation-sign:before {
  content: "\e101";
}
.glyphicon-gift:before {
  content: "\e102";
}
.glyphicon-leaf:before {
  content: "\e103";
}
.glyphicon-fire:before {
  content: "\e104";
}
.glyphicon-eye-open:before {
  content: "\e105";
}
.glyphicon-eye-close:before {
  content: "\e106";
}
.glyphicon-warning-sign:before {
  content: "\e107";
}
.glyphicon-plane:before {
  content: "\e108";
}
.glyphicon-calendar:before {
  content: "\e109";
}
.glyphicon-random:before {
  content: "\e110";
}
.glyphicon-comment:before {
  content: "\e111";
}
.glyphicon-magnet:before {
  content: "\e112";
}
.glyphicon-chevron-up:before {
  content: "\e113";
}
.glyphicon-chevron-down:before {
  content: "\e114";
}
.glyphicon-retweet:before {
  content: "\e115";
}
.glyphicon-shopping-cart:before {
  content: "\e116";
}
.glyphicon-folder-close:before {
  content: "\e117";
}
.glyphicon-folder-open:before {
  content: "\e118";
}
.glyphicon-resize-vertical:before {
  content: "\e119";
}
.glyphicon-resize-horizontal:before {
  content: "\e120";
}
.glyphicon-hdd:before {
  content: "\e121";
}
.glyphicon-bullhorn:before {
  content: "\e122";
}
.glyphicon-bell:before {
  content: "\e123";
}
.glyphicon-certificate:before {
  content: "\e124";
}
.glyphicon-thumbs-up:before {
  content: "\e125";
}
.glyphicon-thumbs-down:before {
  content: "\e126";
}
.glyphicon-hand-right:before {
  content: "\e127";
}
.glyphicon-hand-left:before {
  content: "\e128";
}
.glyphicon-hand-up:before {
  content: "\e129";
}
.glyphicon-hand-down:before {
  content: "\e130";
}
.glyphicon-circle-arrow-right:before {
  content: "\e131";
}
.glyphicon-circle-arrow-left:before {
  content: "\e132";
}
.glyphicon-circle-arrow-up:before {
  content: "\e133";
}
.glyphicon-circle-arrow-down:before {
  content: "\e134";
}
.glyphicon-globe:before {
  content: "\e135";
}
.glyphicon-wrench:before {
  content: "\e136";
}
.glyphicon-tasks:before {
  content: "\e137";
}
.glyphicon-filter:before {
  content: "\e138";
}
.glyphicon-briefcase:before {
  content: "\e139";
}
.glyphicon-fullscreen:before {
  content: "\e140";
}
.glyphicon-dashboard:before {
  content: "\e141";
}
.glyphicon-paperclip:before {
  content: "\e142";
}
.glyphicon-heart-empty:before {
  content: "\e143";
}
.glyphicon-link:before {
  content: "\e144";
}
.glyphicon-phone:before {
  content: "\e145";
}
.glyphicon-pushpin:before {
  content: "\e146";
}
.glyphicon-usd:before {
  content: "\e148";
}
.glyphicon-gbp:before {
  content: "\e149";
}
.glyphicon-sort:before {
  content: "\e150";
}
.glyphicon-sort-by-alphabet:before {
  content: "\e151";
}
.glyphicon-sort-by-alphabet-alt:before {
  content: "\e152";
}
.glyphicon-sort-by-order:before {
  content: "\e153";
}
.glyphicon-sort-by-order-alt:before {
  content: "\e154";
}
.glyphicon-sort-by-attributes:before {
  content: "\e155";
}
.glyphicon-sort-by-attributes-alt:before {
  content: "\e156";
}
.glyphicon-unchecked:before {
  content: "\e157";
}
.glyphicon-expand:before {
  content: "\e158";
}
.glyphicon-collapse-down:before {
  content: "\e159";
}
.glyphicon-collapse-up:before {
  content: "\e160";
}
.glyphicon-log-in:before {
  content: "\e161";
}
.glyphicon-flash:before {
  content: "\e162";
}
.glyphicon-log-out:before {
  content: "\e163";
}
.glyphicon-new-window:before {
  content: "\e164";
}
.glyphicon-record:before {
  content: "\e165";
}
.glyphicon-save:before {
  content: "\e166";
}
.glyphicon-open:before {
  content: "\e167";
}
.glyphicon-saved:before {
  content: "\e168";
}
.glyphicon-import:before {
  content: "\e169";
}
.glyphicon-export:before {
  content: "\e170";
}
.glyphicon-send:before {
  content: "\e171";
}
.glyphicon-floppy-disk:before {
  content: "\e172";
}
.glyphicon-floppy-saved:before {
  content: "\e173";
}
.glyphicon-floppy-remove:before {
  content: "\e174";
}
.glyphicon-floppy-save:before {
  content: "\e175";
}
.glyphicon-floppy-open:before {
  content: "\e176";
}
.glyphicon-credit-card:before {
  content: "\e177";
}
.glyphicon-transfer:before {
  content: "\e178";
}
.glyphicon-cutlery:before {
  content: "\e179";
}
.glyphicon-header:before {
  content: "\e180";
}
.glyphicon-compressed:before {
  content: "\e181";
}
.glyphicon-earphone:before {
  content: "\e182";
}
.glyphicon-phone-alt:before {
  content: "\e183";
}
.glyphicon-tower:before {
  content: "\e184";
}
.glyphicon-stats:before {
  content: "\e185";
}
.glyphicon-sd-video:before {
  content: "\e186";
}
.glyphicon-hd-video:before {
  content: "\e187";
}
.glyphicon-subtitles:before {
  content: "\e188";
}
.glyphicon-sound-stereo:before {
  content: "\e189";
}
.glyphicon-sound-dolby:before {
  content: "\e190";
}
.glyphicon-sound-5-1:before {
  content: "\e191";
}
.glyphicon-sound-6-1:before {
  content: "\e192";
}
.glyphicon-sound-7-1:before {
  content: "\e193";
}
.glyphicon-copyright-mark:before {
  content: "\e194";
}
.glyphicon-registration-mark:before {
  content: "\e195";
}
.glyphicon-cloud-download:before {
  content: "\e197";
}
.glyphicon-cloud-upload:before {
  content: "\e198";
}
.glyphicon-tree-conifer:before {
  content: "\e199";
}
.glyphicon-tree-deciduous:before {
  content: "\e200";
}
.glyphicon-cd:before {
  content: "\e201";
}
.glyphicon-save-file:before {
  content: "\e202";
}
.glyphicon-open-file:before {
  content: "\e203";
}
.glyphicon-level-up:before {
  content: "\e204";
}
.glyphicon-copy:before {
  content: "\e205";
}
.glyphicon-paste:before {
  content: "\e206";
}
.glyphicon-alert:before {
  content: "\e209";
}
.glyphicon-equalizer:before {
  content: "\e210";
}
.glyphicon-king:before {
  content: "\e211";
}
.glyphicon-queen:before {
  content: "\e212";
}
.glyphicon-pawn:before {
  content: "\e213";
}
.glyphicon-bishop:before {
  content: "\e214";
}
.glyphicon-knight:before {
  content: "\e215";
}
.glyphicon-baby-formula:before {
  content: "\e216";
}
.glyphicon-tent:before {
  content: "\26fa";
}
.glyphicon-blackboard:before {
  content: "\e218";
}
.glyphicon-bed:before {
  content: "\e219";
}
.glyphicon-apple:before {
  content: "\f8ff";
}
.glyphicon-erase:before {
  content: "\e221";
}
.glyphicon-hourglass:before {
  content: "\231b";
}
.glyphicon-lamp:before {
  content: "\e223";
}
.glyphicon-duplicate:before {
  content: "\e224";
}
.glyphicon-piggy-bank:before {
  content: "\e225";
}
.glyphicon-scissors:before {
  content: "\e226";
}
.glyphicon-bitcoin:before {
  content: "\e227";
}
.glyphicon-btc:before {
  content: "\e227";
}
.glyphicon-xbt:before {
  content: "\e227";
}
.glyphicon-yen:before {
  content: "\00a5";
}
.glyphicon-jpy:before {
  content: "\00a5";
}
.glyphicon-ruble:before {
  content: "\20bd";
}
.glyphicon-rub:before {
  content: "\20bd";
}
.glyphicon-scale:before {
  content: "\e230";
}
.glyphicon-ice-lolly:before {
  content: "\e231";
}
.glyphicon-ice-lolly-tasted:before {
  content: "\e232";
}
.glyphicon-education:before {
  content: "\e233";
}
.glyphicon-option-horizontal:before {
  content: "\e234";
}
.glyphicon-option-vertical:before {
  content: "\e235";
}
.glyphicon-menu-hamburger:before {
  content: "\e236";
}
.glyphicon-modal-window:before {
  content: "\e237";
}
.glyphicon-oil:before {
  content: "\e238";
}
.glyphicon-grain:before {
  content: "\e239";
}
.glyphicon-sunglasses:before {
  content: "\e240";
}
.glyphicon-text-size:before {
  content: "\e241";
}
.glyphicon-text-color:before {
  content: "\e242";
}
.glyphicon-text-background:before {
  content: "\e243";
}
.glyphicon-object-align-top:before {
  content: "\e244";
}
.glyphicon-object-align-bottom:before {
  content: "\e245";
}
.glyphicon-object-align-horizontal:before {
  content: "\e246";
}
.glyphicon-object-align-left:before {
  content: "\e247";
}
.glyphicon-object-align-vertical:before {
  content: "\e248";
}
.glyphicon-object-align-right:before {
  content: "\e249";
}
.glyphicon-triangle-right:before {
  content: "\e250";
}
.glyphicon-triangle-left:before {
  content: "\e251";
}
.glyphicon-triangle-bottom:before {
  content: "\e252";
}
.glyphicon-triangle-top:before {
  content: "\e253";
}
.glyphicon-console:before {
  content: "\e254";
}
.glyphicon-superscript:before {
  content: "\e255";
}
.glyphicon-subscript:before {
  content: "\e256";
}
.glyphicon-menu-left:before {
  content: "\e257";
}
.glyphicon-menu-right:before {
  content: "\e258";
}
.glyphicon-menu-down:before {
  content: "\e259";
}
.glyphicon-menu-up:before {
  content: "\e260";
}
* {
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  box-sizing: border-box;
}
*:before,
*:after {
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  box-sizing: border-box;
}
html {
  font-size: 10px;
  -webkit-tap-highlight-color: rgba(0, 0, 0, 0);
}
body {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 13px;
  line-height: 1.42857143;
  color: #000;
  background-color: #fff;
}
input,
button,
select,
textarea {
  font-family: inherit;
  font-size: inherit;
  line-height: inherit;
}
a {
  color: #337ab7;
  text-decoration: none;
}
a:hover,
a:focus {
  color: #23527c;
  text-decoration: underline;
}
a:focus {
  outline: 5px auto -webkit-focus-ring-color;
  outline-offset: -2px;
}
figure {
  margin: 0;
}
img {
  vertical-align: middle;
}
.img-responsive,
.thumbnail > img,
.thumbnail a > img,
.carousel-inner > .item > img,
.carousel-inner > .item > a > img {
  display: block;
  max-width: 100%;
  height: auto;
}
.img-rounded {
  border-radius: 3px;
}
.img-thumbnail {
  padding: 4px;
  line-height: 1.42857143;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 2px;
  -webkit-transition: all 0.2s ease-in-out;
  -o-transition: all 0.2s ease-in-out;
  transition: all 0.2s ease-in-out;
  display: inline-block;
  max-width: 100%;
  height: auto;
}
.img-circle {
  border-radius: 50%;
}
hr {
  margin-top: 18px;
  margin-bottom: 18px;
  border: 0;
  border-top: 1px solid #eeeeee;
}
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  margin: -1px;
  padding: 0;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  border: 0;
}
.sr-only-focusable:active,
.sr-only-focusable:focus {
  position: static;
  width: auto;
  height: auto;
  margin: 0;
  overflow: visible;
  clip: auto;
}
[role="button"] {
  cursor: pointer;
}
h1,
h2,
h3,
h4,
h5,
h6,
.h1,
.h2,
.h3,
.h4,
.h5,
.h6 {
  font-family: inherit;
  font-weight: 500;
  line-height: 1.1;
  color: inherit;
}
h1 small,
h2 small,
h3 small,
h4 small,
h5 small,
h6 small,
.h1 small,
.h2 small,
.h3 small,
.h4 small,
.h5 small,
.h6 small,
h1 .small,
h2 .small,
h3 .small,
h4 .small,
h5 .small,
h6 .small,
.h1 .small,
.h2 .small,
.h3 .small,
.h4 .small,
.h5 .small,
.h6 .small {
  font-weight: normal;
  line-height: 1;
  color: #777777;
}
h1,
.h1,
h2,
.h2,
h3,
.h3 {
  margin-top: 18px;
  margin-bottom: 9px;
}
h1 small,
.h1 small,
h2 small,
.h2 small,
h3 small,
.h3 small,
h1 .small,
.h1 .small,
h2 .small,
.h2 .small,
h3 .small,
.h3 .small {
  font-size: 65%;
}
h4,
.h4,
h5,
.h5,
h6,
.h6 {
  margin-top: 9px;
  margin-bottom: 9px;
}
h4 small,
.h4 small,
h5 small,
.h5 small,
h6 small,
.h6 small,
h4 .small,
.h4 .small,
h5 .small,
.h5 .small,
h6 .small,
.h6 .small {
  font-size: 75%;
}
h1,
.h1 {
  font-size: 33px;
}
h2,
.h2 {
  font-size: 27px;
}
h3,
.h3 {
  font-size: 23px;
}
h4,
.h4 {
  font-size: 17px;
}
h5,
.h5 {
  font-size: 13px;
}
h6,
.h6 {
  font-size: 12px;
}
p {
  margin: 0 0 9px;
}
.lead {
  margin-bottom: 18px;
  font-size: 14px;
  font-weight: 300;
  line-height: 1.4;
}
@media (min-width: 768px) {
  .lead {
    font-size: 19.5px;
  }
}
small,
.small {
  font-size: 92%;
}
mark,
.mark {
  background-color: #fcf8e3;
  padding: .2em;
}
.text-left {
  text-align: left;
}
.text-right {
  text-align: right;
}
.text-center {
  text-align: center;
}
.text-justify {
  text-align: justify;
}
.text-nowrap {
  white-space: nowrap;
}
.text-lowercase {
  text-transform: lowercase;
}
.text-uppercase {
  text-transform: uppercase;
}
.text-capitalize {
  text-transform: capitalize;
}
.text-muted {
  color: #777777;
}
.text-primary {
  color: #337ab7;
}
a.text-primary:hover,
a.text-primary:focus {
  color: #286090;
}
.text-success {
  color: #3c763d;
}
a.text-success:hover,
a.text-success:focus {
  color: #2b542c;
}
.text-info {
  color: #31708f;
}
a.text-info:hover,
a.text-info:focus {
  color: #245269;
}
.text-warning {
  color: #8a6d3b;
}
a.text-warning:hover,
a.text-warning:focus {
  color: #66512c;
}
.text-danger {
  color: #a94442;
}
a.text-danger:hover,
a.text-danger:focus {
  color: #843534;
}
.bg-primary {
  color: #fff;
  background-color: #337ab7;
}
a.bg-primary:hover,
a.bg-primary:focus {
  background-color: #286090;
}
.bg-success {
  background-color: #dff0d8;
}
a.bg-success:hover,
a.bg-success:focus {
  background-color: #c1e2b3;
}
.bg-info {
  background-color: #d9edf7;
}
a.bg-info:hover,
a.bg-info:focus {
  background-color: #afd9ee;
}
.bg-warning {
  background-color: #fcf8e3;
}
a.bg-warning:hover,
a.bg-warning:focus {
  background-color: #f7ecb5;
}
.bg-danger {
  background-color: #f2dede;
}
a.bg-danger:hover,
a.bg-danger:focus {
  background-color: #e4b9b9;
}
.page-header {
  padding-bottom: 8px;
  margin: 36px 0 18px;
  border-bottom: 1px solid #eeeeee;
}
ul,
ol {
  margin-top: 0;
  margin-bottom: 9px;
}
ul ul,
ol ul,
ul ol,
ol ol {
  margin-bottom: 0;
}
.list-unstyled {
  padding-left: 0;
  list-style: none;
}
.list-inline {
  padding-left: 0;
  list-style: none;
  margin-left: -5px;
}
.list-inline > li {
  display: inline-block;
  padding-left: 5px;
  padding-right: 5px;
}
dl {
  margin-top: 0;
  margin-bottom: 18px;
}
dt,
dd {
  line-height: 1.42857143;
}
dt {
  font-weight: bold;
}
dd {
  margin-left: 0;
}
@media (min-width: 541px) {
  .dl-horizontal dt {
    float: left;
    width: 160px;
    clear: left;
    text-align: right;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .dl-horizontal dd {
    margin-left: 180px;
  }
}
abbr[title],
abbr[data-original-title] {
  cursor: help;
  border-bottom: 1px dotted #777777;
}
.initialism {
  font-size: 90%;
  text-transform: uppercase;
}
blockquote {
  padding: 9px 18px;
  margin: 0 0 18px;
  font-size: inherit;
  border-left: 5px solid #eeeeee;
}
blockquote p:last-child,
blockquote ul:last-child,
blockquote ol:last-child {
  margin-bottom: 0;
}
blockquote footer,
blockquote small,
blockquote .small {
  display: block;
  font-size: 80%;
  line-height: 1.42857143;
  color: #777777;
}
blockquote footer:before,
blockquote small:before,
blockquote .small:before {
  content: '\2014 \00A0';
}
.blockquote-reverse,
blockquote.pull-right {
  padding-right: 15px;
  padding-left: 0;
  border-right: 5px solid #eeeeee;
  border-left: 0;
  text-align: right;
}
.blockquote-reverse footer:before,
blockquote.pull-right footer:before,
.blockquote-reverse small:before,
blockquote.pull-right small:before,
.blockquote-reverse .small:before,
blockquote.pull-right .small:before {
  content: '';
}
.blockquote-reverse footer:after,
blockquote.pull-right footer:after,
.blockquote-reverse small:after,
blockquote.pull-right small:after,
.blockquote-reverse .small:after,
blockquote.pull-right .small:after {
  content: '\00A0 \2014';
}
address {
  margin-bottom: 18px;
  font-style: normal;
  line-height: 1.42857143;
}
code,
kbd,
pre,
samp {
  font-family: monospace;
}
code {
  padding: 2px 4px;
  font-size: 90%;
  color: #c7254e;
  background-color: #f9f2f4;
  border-radius: 2px;
}
kbd {
  padding: 2px 4px;
  font-size: 90%;
  color: #888;
  background-color: transparent;
  border-radius: 1px;
  box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.25);
}
kbd kbd {
  padding: 0;
  font-size: 100%;
  font-weight: bold;
  box-shadow: none;
}
pre {
  display: block;
  padding: 8.5px;
  margin: 0 0 9px;
  font-size: 12px;
  line-height: 1.42857143;
  word-break: break-all;
  word-wrap: break-word;
  color: #333333;
  background-color: #f5f5f5;
  border: 1px solid #ccc;
  border-radius: 2px;
}
pre code {
  padding: 0;
  font-size: inherit;
  color: inherit;
  white-space: pre-wrap;
  background-color: transparent;
  border-radius: 0;
}
.pre-scrollable {
  max-height: 340px;
  overflow-y: scroll;
}
.container {
  margin-right: auto;
  margin-left: auto;
  padding-left: 0px;
  padding-right: 0px;
}
@media (min-width: 768px) {
  .container {
    width: 768px;
  }
}
@media (min-width: 992px) {
  .container {
    width: 940px;
  }
}
@media (min-width: 1200px) {
  .container {
    width: 1140px;
  }
}
.container-fluid {
  margin-right: auto;
  margin-left: auto;
  padding-left: 0px;
  padding-right: 0px;
}
.row {
  margin-left: 0px;
  margin-right: 0px;
}
.col-xs-1, .col-sm-1, .col-md-1, .col-lg-1, .col-xs-2, .col-sm-2, .col-md-2, .col-lg-2, .col-xs-3, .col-sm-3, .col-md-3, .col-lg-3, .col-xs-4, .col-sm-4, .col-md-4, .col-lg-4, .col-xs-5, .col-sm-5, .col-md-5, .col-lg-5, .col-xs-6, .col-sm-6, .col-md-6, .col-lg-6, .col-xs-7, .col-sm-7, .col-md-7, .col-lg-7, .col-xs-8, .col-sm-8, .col-md-8, .col-lg-8, .col-xs-9, .col-sm-9, .col-md-9, .col-lg-9, .col-xs-10, .col-sm-10, .col-md-10, .col-lg-10, .col-xs-11, .col-sm-11, .col-md-11, .col-lg-11, .col-xs-12, .col-sm-12, .col-md-12, .col-lg-12 {
  position: relative;
  min-height: 1px;
  padding-left: 0px;
  padding-right: 0px;
}
.col-xs-1, .col-xs-2, .col-xs-3, .col-xs-4, .col-xs-5, .col-xs-6, .col-xs-7, .col-xs-8, .col-xs-9, .col-xs-10, .col-xs-11, .col-xs-12 {
  float: left;
}
.col-xs-12 {
  width: 100%;
}
.col-xs-11 {
  width: 91.66666667%;
}
.col-xs-10 {
  width: 83.33333333%;
}
.col-xs-9 {
  width: 75%;
}
.col-xs-8 {
  width: 66.66666667%;
}
.col-xs-7 {
  width: 58.33333333%;
}
.col-xs-6 {
  width: 50%;
}
.col-xs-5 {
  width: 41.66666667%;
}
.col-xs-4 {
  width: 33.33333333%;
}
.col-xs-3 {
  width: 25%;
}
.col-xs-2 {
  width: 16.66666667%;
}
.col-xs-1 {
  width: 8.33333333%;
}
.col-xs-pull-12 {
  right: 100%;
}
.col-xs-pull-11 {
  right: 91.66666667%;
}
.col-xs-pull-10 {
  right: 83.33333333%;
}
.col-xs-pull-9 {
  right: 75%;
}
.col-xs-pull-8 {
  right: 66.66666667%;
}
.col-xs-pull-7 {
  right: 58.33333333%;
}
.col-xs-pull-6 {
  right: 50%;
}
.col-xs-pull-5 {
  right: 41.66666667%;
}
.col-xs-pull-4 {
  right: 33.33333333%;
}
.col-xs-pull-3 {
  right: 25%;
}
.col-xs-pull-2 {
  right: 16.66666667%;
}
.col-xs-pull-1 {
  right: 8.33333333%;
}
.col-xs-pull-0 {
  right: auto;
}
.col-xs-push-12 {
  left: 100%;
}
.col-xs-push-11 {
  left: 91.66666667%;
}
.col-xs-push-10 {
  left: 83.33333333%;
}
.col-xs-push-9 {
  left: 75%;
}
.col-xs-push-8 {
  left: 66.66666667%;
}
.col-xs-push-7 {
  left: 58.33333333%;
}
.col-xs-push-6 {
  left: 50%;
}
.col-xs-push-5 {
  left: 41.66666667%;
}
.col-xs-push-4 {
  left: 33.33333333%;
}
.col-xs-push-3 {
  left: 25%;
}
.col-xs-push-2 {
  left: 16.66666667%;
}
.col-xs-push-1 {
  left: 8.33333333%;
}
.col-xs-push-0 {
  left: auto;
}
.col-xs-offset-12 {
  margin-left: 100%;
}
.col-xs-offset-11 {
  margin-left: 91.66666667%;
}
.col-xs-offset-10 {
  margin-left: 83.33333333%;
}
.col-xs-offset-9 {
  margin-left: 75%;
}
.col-xs-offset-8 {
  margin-left: 66.66666667%;
}
.col-xs-offset-7 {
  margin-left: 58.33333333%;
}
.col-xs-offset-6 {
  margin-left: 50%;
}
.col-xs-offset-5 {
  margin-left: 41.66666667%;
}
.col-xs-offset-4 {
  margin-left: 33.33333333%;
}
.col-xs-offset-3 {
  margin-left: 25%;
}
.col-xs-offset-2 {
  margin-left: 16.66666667%;
}
.col-xs-offset-1 {
  margin-left: 8.33333333%;
}
.col-xs-offset-0 {
  margin-left: 0%;
}
@media (min-width: 768px) {
  .col-sm-1, .col-sm-2, .col-sm-3, .col-sm-4, .col-sm-5, .col-sm-6, .col-sm-7, .col-sm-8, .col-sm-9, .col-sm-10, .col-sm-11, .col-sm-12 {
    float: left;
  }
  .col-sm-12 {
    width: 100%;
  }
  .col-sm-11 {
    width: 91.66666667%;
  }
  .col-sm-10 {
    width: 83.33333333%;
  }
  .col-sm-9 {
    width: 75%;
  }
  .col-sm-8 {
    width: 66.66666667%;
  }
  .col-sm-7 {
    width: 58.33333333%;
  }
  .col-sm-6 {
    width: 50%;
  }
  .col-sm-5 {
    width: 41.66666667%;
  }
  .col-sm-4 {
    width: 33.33333333%;
  }
  .col-sm-3 {
    width: 25%;
  }
  .col-sm-2 {
    width: 16.66666667%;
  }
  .col-sm-1 {
    width: 8.33333333%;
  }
  .col-sm-pull-12 {
    right: 100%;
  }
  .col-sm-pull-11 {
    right: 91.66666667%;
  }
  .col-sm-pull-10 {
    right: 83.33333333%;
  }
  .col-sm-pull-9 {
    right: 75%;
  }
  .col-sm-pull-8 {
    right: 66.66666667%;
  }
  .col-sm-pull-7 {
    right: 58.33333333%;
  }
  .col-sm-pull-6 {
    right: 50%;
  }
  .col-sm-pull-5 {
    right: 41.66666667%;
  }
  .col-sm-pull-4 {
    right: 33.33333333%;
  }
  .col-sm-pull-3 {
    right: 25%;
  }
  .col-sm-pull-2 {
    right: 16.66666667%;
  }
  .col-sm-pull-1 {
    right: 8.33333333%;
  }
  .col-sm-pull-0 {
    right: auto;
  }
  .col-sm-push-12 {
    left: 100%;
  }
  .col-sm-push-11 {
    left: 91.66666667%;
  }
  .col-sm-push-10 {
    left: 83.33333333%;
  }
  .col-sm-push-9 {
    left: 75%;
  }
  .col-sm-push-8 {
    left: 66.66666667%;
  }
  .col-sm-push-7 {
    left: 58.33333333%;
  }
  .col-sm-push-6 {
    left: 50%;
  }
  .col-sm-push-5 {
    left: 41.66666667%;
  }
  .col-sm-push-4 {
    left: 33.33333333%;
  }
  .col-sm-push-3 {
    left: 25%;
  }
  .col-sm-push-2 {
    left: 16.66666667%;
  }
  .col-sm-push-1 {
    left: 8.33333333%;
  }
  .col-sm-push-0 {
    left: auto;
  }
  .col-sm-offset-12 {
    margin-left: 100%;
  }
  .col-sm-offset-11 {
    margin-left: 91.66666667%;
  }
  .col-sm-offset-10 {
    margin-left: 83.33333333%;
  }
  .col-sm-offset-9 {
    margin-left: 75%;
  }
  .col-sm-offset-8 {
    margin-left: 66.66666667%;
  }
  .col-sm-offset-7 {
    margin-left: 58.33333333%;
  }
  .col-sm-offset-6 {
    margin-left: 50%;
  }
  .col-sm-offset-5 {
    margin-left: 41.66666667%;
  }
  .col-sm-offset-4 {
    margin-left: 33.33333333%;
  }
  .col-sm-offset-3 {
    margin-left: 25%;
  }
  .col-sm-offset-2 {
    margin-left: 16.66666667%;
  }
  .col-sm-offset-1 {
    margin-left: 8.33333333%;
  }
  .col-sm-offset-0 {
    margin-left: 0%;
  }
}
@media (min-width: 992px) {
  .col-md-1, .col-md-2, .col-md-3, .col-md-4, .col-md-5, .col-md-6, .col-md-7, .col-md-8, .col-md-9, .col-md-10, .col-md-11, .col-md-12 {
    float: left;
  }
  .col-md-12 {
    width: 100%;
  }
  .col-md-11 {
    width: 91.66666667%;
  }
  .col-md-10 {
    width: 83.33333333%;
  }
  .col-md-9 {
    width: 75%;
  }
  .col-md-8 {
    width: 66.66666667%;
  }
  .col-md-7 {
    width: 58.33333333%;
  }
  .col-md-6 {
    width: 50%;
  }
  .col-md-5 {
    width: 41.66666667%;
  }
  .col-md-4 {
    width: 33.33333333%;
  }
  .col-md-3 {
    width: 25%;
  }
  .col-md-2 {
    width: 16.66666667%;
  }
  .col-md-1 {
    width: 8.33333333%;
  }
  .col-md-pull-12 {
    right: 100%;
  }
  .col-md-pull-11 {
    right: 91.66666667%;
  }
  .col-md-pull-10 {
    right: 83.33333333%;
  }
  .col-md-pull-9 {
    right: 75%;
  }
  .col-md-pull-8 {
    right: 66.66666667%;
  }
  .col-md-pull-7 {
    right: 58.33333333%;
  }
  .col-md-pull-6 {
    right: 50%;
  }
  .col-md-pull-5 {
    right: 41.66666667%;
  }
  .col-md-pull-4 {
    right: 33.33333333%;
  }
  .col-md-pull-3 {
    right: 25%;
  }
  .col-md-pull-2 {
    right: 16.66666667%;
  }
  .col-md-pull-1 {
    right: 8.33333333%;
  }
  .col-md-pull-0 {
    right: auto;
  }
  .col-md-push-12 {
    left: 100%;
  }
  .col-md-push-11 {
    left: 91.66666667%;
  }
  .col-md-push-10 {
    left: 83.33333333%;
  }
  .col-md-push-9 {
    left: 75%;
  }
  .col-md-push-8 {
    left: 66.66666667%;
  }
  .col-md-push-7 {
    left: 58.33333333%;
  }
  .col-md-push-6 {
    left: 50%;
  }
  .col-md-push-5 {
    left: 41.66666667%;
  }
  .col-md-push-4 {
    left: 33.33333333%;
  }
  .col-md-push-3 {
    left: 25%;
  }
  .col-md-push-2 {
    left: 16.66666667%;
  }
  .col-md-push-1 {
    left: 8.33333333%;
  }
  .col-md-push-0 {
    left: auto;
  }
  .col-md-offset-12 {
    margin-left: 100%;
  }
  .col-md-offset-11 {
    margin-left: 91.66666667%;
  }
  .col-md-offset-10 {
    margin-left: 83.33333333%;
  }
  .col-md-offset-9 {
    margin-left: 75%;
  }
  .col-md-offset-8 {
    margin-left: 66.66666667%;
  }
  .col-md-offset-7 {
    margin-left: 58.33333333%;
  }
  .col-md-offset-6 {
    margin-left: 50%;
  }
  .col-md-offset-5 {
    margin-left: 41.66666667%;
  }
  .col-md-offset-4 {
    margin-left: 33.33333333%;
  }
  .col-md-offset-3 {
    margin-left: 25%;
  }
  .col-md-offset-2 {
    margin-left: 16.66666667%;
  }
  .col-md-offset-1 {
    margin-left: 8.33333333%;
  }
  .col-md-offset-0 {
    margin-left: 0%;
  }
}
@media (min-width: 1200px) {
  .col-lg-1, .col-lg-2, .col-lg-3, .col-lg-4, .col-lg-5, .col-lg-6, .col-lg-7, .col-lg-8, .col-lg-9, .col-lg-10, .col-lg-11, .col-lg-12 {
    float: left;
  }
  .col-lg-12 {
    width: 100%;
  }
  .col-lg-11 {
    width: 91.66666667%;
  }
  .col-lg-10 {
    width: 83.33333333%;
  }
  .col-lg-9 {
    width: 75%;
  }
  .col-lg-8 {
    width: 66.66666667%;
  }
  .col-lg-7 {
    width: 58.33333333%;
  }
  .col-lg-6 {
    width: 50%;
  }
  .col-lg-5 {
    width: 41.66666667%;
  }
  .col-lg-4 {
    width: 33.33333333%;
  }
  .col-lg-3 {
    width: 25%;
  }
  .col-lg-2 {
    width: 16.66666667%;
  }
  .col-lg-1 {
    width: 8.33333333%;
  }
  .col-lg-pull-12 {
    right: 100%;
  }
  .col-lg-pull-11 {
    right: 91.66666667%;
  }
  .col-lg-pull-10 {
    right: 83.33333333%;
  }
  .col-lg-pull-9 {
    right: 75%;
  }
  .col-lg-pull-8 {
    right: 66.66666667%;
  }
  .col-lg-pull-7 {
    right: 58.33333333%;
  }
  .col-lg-pull-6 {
    right: 50%;
  }
  .col-lg-pull-5 {
    right: 41.66666667%;
  }
  .col-lg-pull-4 {
    right: 33.33333333%;
  }
  .col-lg-pull-3 {
    right: 25%;
  }
  .col-lg-pull-2 {
    right: 16.66666667%;
  }
  .col-lg-pull-1 {
    right: 8.33333333%;
  }
  .col-lg-pull-0 {
    right: auto;
  }
  .col-lg-push-12 {
    left: 100%;
  }
  .col-lg-push-11 {
    left: 91.66666667%;
  }
  .col-lg-push-10 {
    left: 83.33333333%;
  }
  .col-lg-push-9 {
    left: 75%;
  }
  .col-lg-push-8 {
    left: 66.66666667%;
  }
  .col-lg-push-7 {
    left: 58.33333333%;
  }
  .col-lg-push-6 {
    left: 50%;
  }
  .col-lg-push-5 {
    left: 41.66666667%;
  }
  .col-lg-push-4 {
    left: 33.33333333%;
  }
  .col-lg-push-3 {
    left: 25%;
  }
  .col-lg-push-2 {
    left: 16.66666667%;
  }
  .col-lg-push-1 {
    left: 8.33333333%;
  }
  .col-lg-push-0 {
    left: auto;
  }
  .col-lg-offset-12 {
    margin-left: 100%;
  }
  .col-lg-offset-11 {
    margin-left: 91.66666667%;
  }
  .col-lg-offset-10 {
    margin-left: 83.33333333%;
  }
  .col-lg-offset-9 {
    margin-left: 75%;
  }
  .col-lg-offset-8 {
    margin-left: 66.66666667%;
  }
  .col-lg-offset-7 {
    margin-left: 58.33333333%;
  }
  .col-lg-offset-6 {
    margin-left: 50%;
  }
  .col-lg-offset-5 {
    margin-left: 41.66666667%;
  }
  .col-lg-offset-4 {
    margin-left: 33.33333333%;
  }
  .col-lg-offset-3 {
    margin-left: 25%;
  }
  .col-lg-offset-2 {
    margin-left: 16.66666667%;
  }
  .col-lg-offset-1 {
    margin-left: 8.33333333%;
  }
  .col-lg-offset-0 {
    margin-left: 0%;
  }
}
table {
  background-color: transparent;
}
caption {
  padding-top: 8px;
  padding-bottom: 8px;
  color: #777777;
  text-align: left;
}
th {
  text-align: left;
}
.table {
  width: 100%;
  max-width: 100%;
  margin-bottom: 18px;
}
.table > thead > tr > th,
.table > tbody > tr > th,
.table > tfoot > tr > th,
.table > thead > tr > td,
.table > tbody > tr > td,
.table > tfoot > tr > td {
  padding: 8px;
  line-height: 1.42857143;
  vertical-align: top;
  border-top: 1px solid #ddd;
}
.table > thead > tr > th {
  vertical-align: bottom;
  border-bottom: 2px solid #ddd;
}
.table > caption + thead > tr:first-child > th,
.table > colgroup + thead > tr:first-child > th,
.table > thead:first-child > tr:first-child > th,
.table > caption + thead > tr:first-child > td,
.table > colgroup + thead > tr:first-child > td,
.table > thead:first-child > tr:first-child > td {
  border-top: 0;
}
.table > tbody + tbody {
  border-top: 2px solid #ddd;
}
.table .table {
  background-color: #fff;
}
.table-condensed > thead > tr > th,
.table-condensed > tbody > tr > th,
.table-condensed > tfoot > tr > th,
.table-condensed > thead > tr > td,
.table-condensed > tbody > tr > td,
.table-condensed > tfoot > tr > td {
  padding: 5px;
}
.table-bordered {
  border: 1px solid #ddd;
}
.table-bordered > thead > tr > th,
.table-bordered > tbody > tr > th,
.table-bordered > tfoot > tr > th,
.table-bordered > thead > tr > td,
.table-bordered > tbody > tr > td,
.table-bordered > tfoot > tr > td {
  border: 1px solid #ddd;
}
.table-bordered > thead > tr > th,
.table-bordered > thead > tr > td {
  border-bottom-width: 2px;
}
.table-striped > tbody > tr:nth-of-type(odd) {
  background-color: #f9f9f9;
}
.table-hover > tbody > tr:hover {
  background-color: #f5f5f5;
}
table col[class*="col-"] {
  position: static;
  float: none;
  display: table-column;
}
table td[class*="col-"],
table th[class*="col-"] {
  position: static;
  float: none;
  display: table-cell;
}
.table > thead > tr > td.active,
.table > tbody > tr > td.active,
.table > tfoot > tr > td.active,
.table > thead > tr > th.active,
.table > tbody > tr > th.active,
.table > tfoot > tr > th.active,
.table > thead > tr.active > td,
.table > tbody > tr.active > td,
.table > tfoot > tr.active > td,
.table > thead > tr.active > th,
.table > tbody > tr.active > th,
.table > tfoot > tr.active > th {
  background-color: #f5f5f5;
}
.table-hover > tbody > tr > td.active:hover,
.table-hover > tbody > tr > th.active:hover,
.table-hover > tbody > tr.active:hover > td,
.table-hover > tbody > tr:hover > .active,
.table-hover > tbody > tr.active:hover > th {
  background-color: #e8e8e8;
}
.table > thead > tr > td.success,
.table > tbody > tr > td.success,
.table > tfoot > tr > td.success,
.table > thead > tr > th.success,
.table > tbody > tr > th.success,
.table > tfoot > tr > th.success,
.table > thead > tr.success > td,
.table > tbody > tr.success > td,
.table > tfoot > tr.success > td,
.table > thead > tr.success > th,
.table > tbody > tr.success > th,
.table > tfoot > tr.success > th {
  background-color: #dff0d8;
}
.table-hover > tbody > tr > td.success:hover,
.table-hover > tbody > tr > th.success:hover,
.table-hover > tbody > tr.success:hover > td,
.table-hover > tbody > tr:hover > .success,
.table-hover > tbody > tr.success:hover > th {
  background-color: #d0e9c6;
}
.table > thead > tr > td.info,
.table > tbody > tr > td.info,
.table > tfoot > tr > td.info,
.table > thead > tr > th.info,
.table > tbody > tr > th.info,
.table > tfoot > tr > th.info,
.table > thead > tr.info > td,
.table > tbody > tr.info > td,
.table > tfoot > tr.info > td,
.table > thead > tr.info > th,
.table > tbody > tr.info > th,
.table > tfoot > tr.info > th {
  background-color: #d9edf7;
}
.table-hover > tbody > tr > td.info:hover,
.table-hover > tbody > tr > th.info:hover,
.table-hover > tbody > tr.info:hover > td,
.table-hover > tbody > tr:hover > .info,
.table-hover > tbody > tr.info:hover > th {
  background-color: #c4e3f3;
}
.table > thead > tr > td.warning,
.table > tbody > tr > td.warning,
.table > tfoot > tr > td.warning,
.table > thead > tr > th.warning,
.table > tbody > tr > th.warning,
.table > tfoot > tr > th.warning,
.table > thead > tr.warning > td,
.table > tbody > tr.warning > td,
.table > tfoot > tr.warning > td,
.table > thead > tr.warning > th,
.table > tbody > tr.warning > th,
.table > tfoot > tr.warning > th {
  background-color: #fcf8e3;
}
.table-hover > tbody > tr > td.warning:hover,
.table-hover > tbody > tr > th.warning:hover,
.table-hover > tbody > tr.warning:hover > td,
.table-hover > tbody > tr:hover > .warning,
.table-hover > tbody > tr.warning:hover > th {
  background-color: #faf2cc;
}
.table > thead > tr > td.danger,
.table > tbody > tr > td.danger,
.table > tfoot > tr > td.danger,
.table > thead > tr > th.danger,
.table > tbody > tr > th.danger,
.table > tfoot > tr > th.danger,
.table > thead > tr.danger > td,
.table > tbody > tr.danger > td,
.table > tfoot > tr.danger > td,
.table > thead > tr.danger > th,
.table > tbody > tr.danger > th,
.table > tfoot > tr.danger > th {
  background-color: #f2dede;
}
.table-hover > tbody > tr > td.danger:hover,
.table-hover > tbody > tr > th.danger:hover,
.table-hover > tbody > tr.danger:hover > td,
.table-hover > tbody > tr:hover > .danger,
.table-hover > tbody > tr.danger:hover > th {
  background-color: #ebcccc;
}
.table-responsive {
  overflow-x: auto;
  min-height: 0.01%;
}
@media screen and (max-width: 767px) {
  .table-responsive {
    width: 100%;
    margin-bottom: 13.5px;
    overflow-y: hidden;
    -ms-overflow-style: -ms-autohiding-scrollbar;
    border: 1px solid #ddd;
  }
  .table-responsive > .table {
    margin-bottom: 0;
  }
  .table-responsive > .table > thead > tr > th,
  .table-responsive > .table > tbody > tr > th,
  .table-responsive > .table > tfoot > tr > th,
  .table-responsive > .table > thead > tr > td,
  .table-responsive > .table > tbody > tr > td,
  .table-responsive > .table > tfoot > tr > td {
    white-space: nowrap;
  }
  .table-responsive > .table-bordered {
    border: 0;
  }
  .table-responsive > .table-bordered > thead > tr > th:first-child,
  .table-responsive > .table-bordered > tbody > tr > th:first-child,
  .table-responsive > .table-bordered > tfoot > tr > th:first-child,
  .table-responsive > .table-bordered > thead > tr > td:first-child,
  .table-responsive > .table-bordered > tbody > tr > td:first-child,
  .table-responsive > .table-bordered > tfoot > tr > td:first-child {
    border-left: 0;
  }
  .table-responsive > .table-bordered > thead > tr > th:last-child,
  .table-responsive > .table-bordered > tbody > tr > th:last-child,
  .table-responsive > .table-bordered > tfoot > tr > th:last-child,
  .table-responsive > .table-bordered > thead > tr > td:last-child,
  .table-responsive > .table-bordered > tbody > tr > td:last-child,
  .table-responsive > .table-bordered > tfoot > tr > td:last-child {
    border-right: 0;
  }
  .table-responsive > .table-bordered > tbody > tr:last-child > th,
  .table-responsive > .table-bordered > tfoot > tr:last-child > th,
  .table-responsive > .table-bordered > tbody > tr:last-child > td,
  .table-responsive > .table-bordered > tfoot > tr:last-child > td {
    border-bottom: 0;
  }
}
fieldset {
  padding: 0;
  margin: 0;
  border: 0;
  min-width: 0;
}
legend {
  display: block;
  width: 100%;
  padding: 0;
  margin-bottom: 18px;
  font-size: 19.5px;
  line-height: inherit;
  color: #333333;
  border: 0;
  border-bottom: 1px solid #e5e5e5;
}
label {
  display: inline-block;
  max-width: 100%;
  margin-bottom: 5px;
  font-weight: bold;
}
input[type="search"] {
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  box-sizing: border-box;
}
input[type="radio"],
input[type="checkbox"] {
  margin: 4px 0 0;
  margin-top: 1px \9;
  line-height: normal;
}
input[type="file"] {
  display: block;
}
input[type="range"] {
  display: block;
  width: 100%;
}
select[multiple],
select[size] {
  height: auto;
}
input[type="file"]:focus,
input[type="radio"]:focus,
input[type="checkbox"]:focus {
  outline: 5px auto -webkit-focus-ring-color;
  outline-offset: -2px;
}
output {
  display: block;
  padding-top: 7px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
}
.form-control {
  display: block;
  width: 100%;
  height: 32px;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
  background-color: #fff;
  background-image: none;
  border: 1px solid #ccc;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  -webkit-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  -o-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
}
.form-control:focus {
  border-color: #66afe9;
  outline: 0;
  -webkit-box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
  box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
}
.form-control::-moz-placeholder {
  color: #999;
  opacity: 1;
}
.form-control:-ms-input-placeholder {
  color: #999;
}
.form-control::-webkit-input-placeholder {
  color: #999;
}
.form-control::-ms-expand {
  border: 0;
  background-color: transparent;
}
.form-control[disabled],
.form-control[readonly],
fieldset[disabled] .form-control {
  background-color: #eeeeee;
  opacity: 1;
}
.form-control[disabled],
fieldset[disabled] .form-control {
  cursor: not-allowed;
}
textarea.form-control {
  height: auto;
}
input[type="search"] {
  -webkit-appearance: none;
}
@media screen and (-webkit-min-device-pixel-ratio: 0) {
  input[type="date"].form-control,
  input[type="time"].form-control,
  input[type="datetime-local"].form-control,
  input[type="month"].form-control {
    line-height: 32px;
  }
  input[type="date"].input-sm,
  input[type="time"].input-sm,
  input[type="datetime-local"].input-sm,
  input[type="month"].input-sm,
  .input-group-sm input[type="date"],
  .input-group-sm input[type="time"],
  .input-group-sm input[type="datetime-local"],
  .input-group-sm input[type="month"] {
    line-height: 30px;
  }
  input[type="date"].input-lg,
  input[type="time"].input-lg,
  input[type="datetime-local"].input-lg,
  input[type="month"].input-lg,
  .input-group-lg input[type="date"],
  .input-group-lg input[type="time"],
  .input-group-lg input[type="datetime-local"],
  .input-group-lg input[type="month"] {
    line-height: 45px;
  }
}
.form-group {
  margin-bottom: 15px;
}
.radio,
.checkbox {
  position: relative;
  display: block;
  margin-top: 10px;
  margin-bottom: 10px;
}
.radio label,
.checkbox label {
  min-height: 18px;
  padding-left: 20px;
  margin-bottom: 0;
  font-weight: normal;
  cursor: pointer;
}
.radio input[type="radio"],
.radio-inline input[type="radio"],
.checkbox input[type="checkbox"],
.checkbox-inline input[type="checkbox"] {
  position: absolute;
  margin-left: -20px;
  margin-top: 4px \9;
}
.radio + .radio,
.checkbox + .checkbox {
  margin-top: -5px;
}
.radio-inline,
.checkbox-inline {
  position: relative;
  display: inline-block;
  padding-left: 20px;
  margin-bottom: 0;
  vertical-align: middle;
  font-weight: normal;
  cursor: pointer;
}
.radio-inline + .radio-inline,
.checkbox-inline + .checkbox-inline {
  margin-top: 0;
  margin-left: 10px;
}
input[type="radio"][disabled],
input[type="checkbox"][disabled],
input[type="radio"].disabled,
input[type="checkbox"].disabled,
fieldset[disabled] input[type="radio"],
fieldset[disabled] input[type="checkbox"] {
  cursor: not-allowed;
}
.radio-inline.disabled,
.checkbox-inline.disabled,
fieldset[disabled] .radio-inline,
fieldset[disabled] .checkbox-inline {
  cursor: not-allowed;
}
.radio.disabled label,
.checkbox.disabled label,
fieldset[disabled] .radio label,
fieldset[disabled] .checkbox label {
  cursor: not-allowed;
}
.form-control-static {
  padding-top: 7px;
  padding-bottom: 7px;
  margin-bottom: 0;
  min-height: 31px;
}
.form-control-static.input-lg,
.form-control-static.input-sm {
  padding-left: 0;
  padding-right: 0;
}
.input-sm {
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
select.input-sm {
  height: 30px;
  line-height: 30px;
}
textarea.input-sm,
select[multiple].input-sm {
  height: auto;
}
.form-group-sm .form-control {
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
.form-group-sm select.form-control {
  height: 30px;
  line-height: 30px;
}
.form-group-sm textarea.form-control,
.form-group-sm select[multiple].form-control {
  height: auto;
}
.form-group-sm .form-control-static {
  height: 30px;
  min-height: 30px;
  padding: 6px 10px;
  font-size: 12px;
  line-height: 1.5;
}
.input-lg {
  height: 45px;
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
select.input-lg {
  height: 45px;
  line-height: 45px;
}
textarea.input-lg,
select[multiple].input-lg {
  height: auto;
}
.form-group-lg .form-control {
  height: 45px;
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
.form-group-lg select.form-control {
  height: 45px;
  line-height: 45px;
}
.form-group-lg textarea.form-control,
.form-group-lg select[multiple].form-control {
  height: auto;
}
.form-group-lg .form-control-static {
  height: 45px;
  min-height: 35px;
  padding: 11px 16px;
  font-size: 17px;
  line-height: 1.3333333;
}
.has-feedback {
  position: relative;
}
.has-feedback .form-control {
  padding-right: 40px;
}
.form-control-feedback {
  position: absolute;
  top: 0;
  right: 0;
  z-index: 2;
  display: block;
  width: 32px;
  height: 32px;
  line-height: 32px;
  text-align: center;
  pointer-events: none;
}
.input-lg + .form-control-feedback,
.input-group-lg + .form-control-feedback,
.form-group-lg .form-control + .form-control-feedback {
  width: 45px;
  height: 45px;
  line-height: 45px;
}
.input-sm + .form-control-feedback,
.input-group-sm + .form-control-feedback,
.form-group-sm .form-control + .form-control-feedback {
  width: 30px;
  height: 30px;
  line-height: 30px;
}
.has-success .help-block,
.has-success .control-label,
.has-success .radio,
.has-success .checkbox,
.has-success .radio-inline,
.has-success .checkbox-inline,
.has-success.radio label,
.has-success.checkbox label,
.has-success.radio-inline label,
.has-success.checkbox-inline label {
  color: #3c763d;
}
.has-success .form-control {
  border-color: #3c763d;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
}
.has-success .form-control:focus {
  border-color: #2b542c;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #67b168;
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #67b168;
}
.has-success .input-group-addon {
  color: #3c763d;
  border-color: #3c763d;
  background-color: #dff0d8;
}
.has-success .form-control-feedback {
  color: #3c763d;
}
.has-warning .help-block,
.has-warning .control-label,
.has-warning .radio,
.has-warning .checkbox,
.has-warning .radio-inline,
.has-warning .checkbox-inline,
.has-warning.radio label,
.has-warning.checkbox label,
.has-warning.radio-inline label,
.has-warning.checkbox-inline label {
  color: #8a6d3b;
}
.has-warning .form-control {
  border-color: #8a6d3b;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
}
.has-warning .form-control:focus {
  border-color: #66512c;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #c0a16b;
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #c0a16b;
}
.has-warning .input-group-addon {
  color: #8a6d3b;
  border-color: #8a6d3b;
  background-color: #fcf8e3;
}
.has-warning .form-control-feedback {
  color: #8a6d3b;
}
.has-error .help-block,
.has-error .control-label,
.has-error .radio,
.has-error .checkbox,
.has-error .radio-inline,
.has-error .checkbox-inline,
.has-error.radio label,
.has-error.checkbox label,
.has-error.radio-inline label,
.has-error.checkbox-inline label {
  color: #a94442;
}
.has-error .form-control {
  border-color: #a94442;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
}
.has-error .form-control:focus {
  border-color: #843534;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #ce8483;
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #ce8483;
}
.has-error .input-group-addon {
  color: #a94442;
  border-color: #a94442;
  background-color: #f2dede;
}
.has-error .form-control-feedback {
  color: #a94442;
}
.has-feedback label ~ .form-control-feedback {
  top: 23px;
}
.has-feedback label.sr-only ~ .form-control-feedback {
  top: 0;
}
.help-block {
  display: block;
  margin-top: 5px;
  margin-bottom: 10px;
  color: #404040;
}
@media (min-width: 768px) {
  .form-inline .form-group {
    display: inline-block;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .form-inline .form-control {
    display: inline-block;
    width: auto;
    vertical-align: middle;
  }
  .form-inline .form-control-static {
    display: inline-block;
  }
  .form-inline .input-group {
    display: inline-table;
    vertical-align: middle;
  }
  .form-inline .input-group .input-group-addon,
  .form-inline .input-group .input-group-btn,
  .form-inline .input-group .form-control {
    width: auto;
  }
  .form-inline .input-group > .form-control {
    width: 100%;
  }
  .form-inline .control-label {
    margin-bottom: 0;
    vertical-align: middle;
  }
  .form-inline .radio,
  .form-inline .checkbox {
    display: inline-block;
    margin-top: 0;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .form-inline .radio label,
  .form-inline .checkbox label {
    padding-left: 0;
  }
  .form-inline .radio input[type="radio"],
  .form-inline .checkbox input[type="checkbox"] {
    position: relative;
    margin-left: 0;
  }
  .form-inline .has-feedback .form-control-feedback {
    top: 0;
  }
}
.form-horizontal .radio,
.form-horizontal .checkbox,
.form-horizontal .radio-inline,
.form-horizontal .checkbox-inline {
  margin-top: 0;
  margin-bottom: 0;
  padding-top: 7px;
}
.form-horizontal .radio,
.form-horizontal .checkbox {
  min-height: 25px;
}
.form-horizontal .form-group {
  margin-left: 0px;
  margin-right: 0px;
}
@media (min-width: 768px) {
  .form-horizontal .control-label {
    text-align: right;
    margin-bottom: 0;
    padding-top: 7px;
  }
}
.form-horizontal .has-feedback .form-control-feedback {
  right: 0px;
}
@media (min-width: 768px) {
  .form-horizontal .form-group-lg .control-label {
    padding-top: 11px;
    font-size: 17px;
  }
}
@media (min-width: 768px) {
  .form-horizontal .form-group-sm .control-label {
    padding-top: 6px;
    font-size: 12px;
  }
}
.btn {
  display: inline-block;
  margin-bottom: 0;
  font-weight: normal;
  text-align: center;
  vertical-align: middle;
  touch-action: manipulation;
  cursor: pointer;
  background-image: none;
  border: 1px solid transparent;
  white-space: nowrap;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  border-radius: 2px;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}
.btn:focus,
.btn:active:focus,
.btn.active:focus,
.btn.focus,
.btn:active.focus,
.btn.active.focus {
  outline: 5px auto -webkit-focus-ring-color;
  outline-offset: -2px;
}
.btn:hover,
.btn:focus,
.btn.focus {
  color: #333;
  text-decoration: none;
}
.btn:active,
.btn.active {
  outline: 0;
  background-image: none;
  -webkit-box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
  box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
}
.btn.disabled,
.btn[disabled],
fieldset[disabled] .btn {
  cursor: not-allowed;
  opacity: 0.65;
  filter: alpha(opacity=65);
  -webkit-box-shadow: none;
  box-shadow: none;
}
a.btn.disabled,
fieldset[disabled] a.btn {
  pointer-events: none;
}
.btn-default {
  color: #333;
  background-color: #fff;
  border-color: #ccc;
}
.btn-default:focus,
.btn-default.focus {
  color: #333;
  background-color: #e6e6e6;
  border-color: #8c8c8c;
}
.btn-default:hover {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.btn-default:active,
.btn-default.active,
.open > .dropdown-toggle.btn-default {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.btn-default:active:hover,
.btn-default.active:hover,
.open > .dropdown-toggle.btn-default:hover,
.btn-default:active:focus,
.btn-default.active:focus,
.open > .dropdown-toggle.btn-default:focus,
.btn-default:active.focus,
.btn-default.active.focus,
.open > .dropdown-toggle.btn-default.focus {
  color: #333;
  background-color: #d4d4d4;
  border-color: #8c8c8c;
}
.btn-default:active,
.btn-default.active,
.open > .dropdown-toggle.btn-default {
  background-image: none;
}
.btn-default.disabled:hover,
.btn-default[disabled]:hover,
fieldset[disabled] .btn-default:hover,
.btn-default.disabled:focus,
.btn-default[disabled]:focus,
fieldset[disabled] .btn-default:focus,
.btn-default.disabled.focus,
.btn-default[disabled].focus,
fieldset[disabled] .btn-default.focus {
  background-color: #fff;
  border-color: #ccc;
}
.btn-default .badge {
  color: #fff;
  background-color: #333;
}
.btn-primary {
  color: #fff;
  background-color: #337ab7;
  border-color: #2e6da4;
}
.btn-primary:focus,
.btn-primary.focus {
  color: #fff;
  background-color: #286090;
  border-color: #122b40;
}
.btn-primary:hover {
  color: #fff;
  background-color: #286090;
  border-color: #204d74;
}
.btn-primary:active,
.btn-primary.active,
.open > .dropdown-toggle.btn-primary {
  color: #fff;
  background-color: #286090;
  border-color: #204d74;
}
.btn-primary:active:hover,
.btn-primary.active:hover,
.open > .dropdown-toggle.btn-primary:hover,
.btn-primary:active:focus,
.btn-primary.active:focus,
.open > .dropdown-toggle.btn-primary:focus,
.btn-primary:active.focus,
.btn-primary.active.focus,
.open > .dropdown-toggle.btn-primary.focus {
  color: #fff;
  background-color: #204d74;
  border-color: #122b40;
}
.btn-primary:active,
.btn-primary.active,
.open > .dropdown-toggle.btn-primary {
  background-image: none;
}
.btn-primary.disabled:hover,
.btn-primary[disabled]:hover,
fieldset[disabled] .btn-primary:hover,
.btn-primary.disabled:focus,
.btn-primary[disabled]:focus,
fieldset[disabled] .btn-primary:focus,
.btn-primary.disabled.focus,
.btn-primary[disabled].focus,
fieldset[disabled] .btn-primary.focus {
  background-color: #337ab7;
  border-color: #2e6da4;
}
.btn-primary .badge {
  color: #337ab7;
  background-color: #fff;
}
.btn-success {
  color: #fff;
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.btn-success:focus,
.btn-success.focus {
  color: #fff;
  background-color: #449d44;
  border-color: #255625;
}
.btn-success:hover {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.btn-success:active,
.btn-success.active,
.open > .dropdown-toggle.btn-success {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.btn-success:active:hover,
.btn-success.active:hover,
.open > .dropdown-toggle.btn-success:hover,
.btn-success:active:focus,
.btn-success.active:focus,
.open > .dropdown-toggle.btn-success:focus,
.btn-success:active.focus,
.btn-success.active.focus,
.open > .dropdown-toggle.btn-success.focus {
  color: #fff;
  background-color: #398439;
  border-color: #255625;
}
.btn-success:active,
.btn-success.active,
.open > .dropdown-toggle.btn-success {
  background-image: none;
}
.btn-success.disabled:hover,
.btn-success[disabled]:hover,
fieldset[disabled] .btn-success:hover,
.btn-success.disabled:focus,
.btn-success[disabled]:focus,
fieldset[disabled] .btn-success:focus,
.btn-success.disabled.focus,
.btn-success[disabled].focus,
fieldset[disabled] .btn-success.focus {
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.btn-success .badge {
  color: #5cb85c;
  background-color: #fff;
}
.btn-info {
  color: #fff;
  background-color: #5bc0de;
  border-color: #46b8da;
}
.btn-info:focus,
.btn-info.focus {
  color: #fff;
  background-color: #31b0d5;
  border-color: #1b6d85;
}
.btn-info:hover {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.btn-info:active,
.btn-info.active,
.open > .dropdown-toggle.btn-info {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.btn-info:active:hover,
.btn-info.active:hover,
.open > .dropdown-toggle.btn-info:hover,
.btn-info:active:focus,
.btn-info.active:focus,
.open > .dropdown-toggle.btn-info:focus,
.btn-info:active.focus,
.btn-info.active.focus,
.open > .dropdown-toggle.btn-info.focus {
  color: #fff;
  background-color: #269abc;
  border-color: #1b6d85;
}
.btn-info:active,
.btn-info.active,
.open > .dropdown-toggle.btn-info {
  background-image: none;
}
.btn-info.disabled:hover,
.btn-info[disabled]:hover,
fieldset[disabled] .btn-info:hover,
.btn-info.disabled:focus,
.btn-info[disabled]:focus,
fieldset[disabled] .btn-info:focus,
.btn-info.disabled.focus,
.btn-info[disabled].focus,
fieldset[disabled] .btn-info.focus {
  background-color: #5bc0de;
  border-color: #46b8da;
}
.btn-info .badge {
  color: #5bc0de;
  background-color: #fff;
}
.btn-warning {
  color: #fff;
  background-color: #f0ad4e;
  border-color: #eea236;
}
.btn-warning:focus,
.btn-warning.focus {
  color: #fff;
  background-color: #ec971f;
  border-color: #985f0d;
}
.btn-warning:hover {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.btn-warning:active,
.btn-warning.active,
.open > .dropdown-toggle.btn-warning {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.btn-warning:active:hover,
.btn-warning.active:hover,
.open > .dropdown-toggle.btn-warning:hover,
.btn-warning:active:focus,
.btn-warning.active:focus,
.open > .dropdown-toggle.btn-warning:focus,
.btn-warning:active.focus,
.btn-warning.active.focus,
.open > .dropdown-toggle.btn-warning.focus {
  color: #fff;
  background-color: #d58512;
  border-color: #985f0d;
}
.btn-warning:active,
.btn-warning.active,
.open > .dropdown-toggle.btn-warning {
  background-image: none;
}
.btn-warning.disabled:hover,
.btn-warning[disabled]:hover,
fieldset[disabled] .btn-warning:hover,
.btn-warning.disabled:focus,
.btn-warning[disabled]:focus,
fieldset[disabled] .btn-warning:focus,
.btn-warning.disabled.focus,
.btn-warning[disabled].focus,
fieldset[disabled] .btn-warning.focus {
  background-color: #f0ad4e;
  border-color: #eea236;
}
.btn-warning .badge {
  color: #f0ad4e;
  background-color: #fff;
}
.btn-danger {
  color: #fff;
  background-color: #d9534f;
  border-color: #d43f3a;
}
.btn-danger:focus,
.btn-danger.focus {
  color: #fff;
  background-color: #c9302c;
  border-color: #761c19;
}
.btn-danger:hover {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.btn-danger:active,
.btn-danger.active,
.open > .dropdown-toggle.btn-danger {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.btn-danger:active:hover,
.btn-danger.active:hover,
.open > .dropdown-toggle.btn-danger:hover,
.btn-danger:active:focus,
.btn-danger.active:focus,
.open > .dropdown-toggle.btn-danger:focus,
.btn-danger:active.focus,
.btn-danger.active.focus,
.open > .dropdown-toggle.btn-danger.focus {
  color: #fff;
  background-color: #ac2925;
  border-color: #761c19;
}
.btn-danger:active,
.btn-danger.active,
.open > .dropdown-toggle.btn-danger {
  background-image: none;
}
.btn-danger.disabled:hover,
.btn-danger[disabled]:hover,
fieldset[disabled] .btn-danger:hover,
.btn-danger.disabled:focus,
.btn-danger[disabled]:focus,
fieldset[disabled] .btn-danger:focus,
.btn-danger.disabled.focus,
.btn-danger[disabled].focus,
fieldset[disabled] .btn-danger.focus {
  background-color: #d9534f;
  border-color: #d43f3a;
}
.btn-danger .badge {
  color: #d9534f;
  background-color: #fff;
}
.btn-link {
  color: #337ab7;
  font-weight: normal;
  border-radius: 0;
}
.btn-link,
.btn-link:active,
.btn-link.active,
.btn-link[disabled],
fieldset[disabled] .btn-link {
  background-color: transparent;
  -webkit-box-shadow: none;
  box-shadow: none;
}
.btn-link,
.btn-link:hover,
.btn-link:focus,
.btn-link:active {
  border-color: transparent;
}
.btn-link:hover,
.btn-link:focus {
  color: #23527c;
  text-decoration: underline;
  background-color: transparent;
}
.btn-link[disabled]:hover,
fieldset[disabled] .btn-link:hover,
.btn-link[disabled]:focus,
fieldset[disabled] .btn-link:focus {
  color: #777777;
  text-decoration: none;
}
.btn-lg,
.btn-group-lg > .btn {
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
.btn-sm,
.btn-group-sm > .btn {
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
.btn-xs,
.btn-group-xs > .btn {
  padding: 1px 5px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
.btn-block {
  display: block;
  width: 100%;
}
.btn-block + .btn-block {
  margin-top: 5px;
}
input[type="submit"].btn-block,
input[type="reset"].btn-block,
input[type="button"].btn-block {
  width: 100%;
}
.fade {
  opacity: 0;
  -webkit-transition: opacity 0.15s linear;
  -o-transition: opacity 0.15s linear;
  transition: opacity 0.15s linear;
}
.fade.in {
  opacity: 1;
}
.collapse {
  display: none;
}
.collapse.in {
  display: block;
}
tr.collapse.in {
  display: table-row;
}
tbody.collapse.in {
  display: table-row-group;
}
.collapsing {
  position: relative;
  height: 0;
  overflow: hidden;
  -webkit-transition-property: height, visibility;
  transition-property: height, visibility;
  -webkit-transition-duration: 0.35s;
  transition-duration: 0.35s;
  -webkit-transition-timing-function: ease;
  transition-timing-function: ease;
}
.caret {
  display: inline-block;
  width: 0;
  height: 0;
  margin-left: 2px;
  vertical-align: middle;
  border-top: 4px dashed;
  border-top: 4px solid \9;
  border-right: 4px solid transparent;
  border-left: 4px solid transparent;
}
.dropup,
.dropdown {
  position: relative;
}
.dropdown-toggle:focus {
  outline: 0;
}
.dropdown-menu {
  position: absolute;
  top: 100%;
  left: 0;
  z-index: 1000;
  display: none;
  float: left;
  min-width: 160px;
  padding: 5px 0;
  margin: 2px 0 0;
  list-style: none;
  font-size: 13px;
  text-align: left;
  background-color: #fff;
  border: 1px solid #ccc;
  border: 1px solid rgba(0, 0, 0, 0.15);
  border-radius: 2px;
  -webkit-box-shadow: 0 6px 12px rgba(0, 0, 0, 0.175);
  box-shadow: 0 6px 12px rgba(0, 0, 0, 0.175);
  background-clip: padding-box;
}
.dropdown-menu.pull-right {
  right: 0;
  left: auto;
}
.dropdown-menu .divider {
  height: 1px;
  margin: 8px 0;
  overflow: hidden;
  background-color: #e5e5e5;
}
.dropdown-menu > li > a {
  display: block;
  padding: 3px 20px;
  clear: both;
  font-weight: normal;
  line-height: 1.42857143;
  color: #333333;
  white-space: nowrap;
}
.dropdown-menu > li > a:hover,
.dropdown-menu > li > a:focus {
  text-decoration: none;
  color: #262626;
  background-color: #f5f5f5;
}
.dropdown-menu > .active > a,
.dropdown-menu > .active > a:hover,
.dropdown-menu > .active > a:focus {
  color: #fff;
  text-decoration: none;
  outline: 0;
  background-color: #337ab7;
}
.dropdown-menu > .disabled > a,
.dropdown-menu > .disabled > a:hover,
.dropdown-menu > .disabled > a:focus {
  color: #777777;
}
.dropdown-menu > .disabled > a:hover,
.dropdown-menu > .disabled > a:focus {
  text-decoration: none;
  background-color: transparent;
  background-image: none;
  filter: progid:DXImageTransform.Microsoft.gradient(enabled = false);
  cursor: not-allowed;
}
.open > .dropdown-menu {
  display: block;
}
.open > a {
  outline: 0;
}
.dropdown-menu-right {
  left: auto;
  right: 0;
}
.dropdown-menu-left {
  left: 0;
  right: auto;
}
.dropdown-header {
  display: block;
  padding: 3px 20px;
  font-size: 12px;
  line-height: 1.42857143;
  color: #777777;
  white-space: nowrap;
}
.dropdown-backdrop {
  position: fixed;
  left: 0;
  right: 0;
  bottom: 0;
  top: 0;
  z-index: 990;
}
.pull-right > .dropdown-menu {
  right: 0;
  left: auto;
}
.dropup .caret,
.navbar-fixed-bottom .dropdown .caret {
  border-top: 0;
  border-bottom: 4px dashed;
  border-bottom: 4px solid \9;
  content: "";
}
.dropup .dropdown-menu,
.navbar-fixed-bottom .dropdown .dropdown-menu {
  top: auto;
  bottom: 100%;
  margin-bottom: 2px;
}
@media (min-width: 541px) {
  .navbar-right .dropdown-menu {
    left: auto;
    right: 0;
  }
  .navbar-right .dropdown-menu-left {
    left: 0;
    right: auto;
  }
}
.btn-group,
.btn-group-vertical {
  position: relative;
  display: inline-block;
  vertical-align: middle;
}
.btn-group > .btn,
.btn-group-vertical > .btn {
  position: relative;
  float: left;
}
.btn-group > .btn:hover,
.btn-group-vertical > .btn:hover,
.btn-group > .btn:focus,
.btn-group-vertical > .btn:focus,
.btn-group > .btn:active,
.btn-group-vertical > .btn:active,
.btn-group > .btn.active,
.btn-group-vertical > .btn.active {
  z-index: 2;
}
.btn-group .btn + .btn,
.btn-group .btn + .btn-group,
.btn-group .btn-group + .btn,
.btn-group .btn-group + .btn-group {
  margin-left: -1px;
}
.btn-toolbar {
  margin-left: -5px;
}
.btn-toolbar .btn,
.btn-toolbar .btn-group,
.btn-toolbar .input-group {
  float: left;
}
.btn-toolbar > .btn,
.btn-toolbar > .btn-group,
.btn-toolbar > .input-group {
  margin-left: 5px;
}
.btn-group > .btn:not(:first-child):not(:last-child):not(.dropdown-toggle) {
  border-radius: 0;
}
.btn-group > .btn:first-child {
  margin-left: 0;
}
.btn-group > .btn:first-child:not(:last-child):not(.dropdown-toggle) {
  border-bottom-right-radius: 0;
  border-top-right-radius: 0;
}
.btn-group > .btn:last-child:not(:first-child),
.btn-group > .dropdown-toggle:not(:first-child) {
  border-bottom-left-radius: 0;
  border-top-left-radius: 0;
}
.btn-group > .btn-group {
  float: left;
}
.btn-group > .btn-group:not(:first-child):not(:last-child) > .btn {
  border-radius: 0;
}
.btn-group > .btn-group:first-child:not(:last-child) > .btn:last-child,
.btn-group > .btn-group:first-child:not(:last-child) > .dropdown-toggle {
  border-bottom-right-radius: 0;
  border-top-right-radius: 0;
}
.btn-group > .btn-group:last-child:not(:first-child) > .btn:first-child {
  border-bottom-left-radius: 0;
  border-top-left-radius: 0;
}
.btn-group .dropdown-toggle:active,
.btn-group.open .dropdown-toggle {
  outline: 0;
}
.btn-group > .btn + .dropdown-toggle {
  padding-left: 8px;
  padding-right: 8px;
}
.btn-group > .btn-lg + .dropdown-toggle {
  padding-left: 12px;
  padding-right: 12px;
}
.btn-group.open .dropdown-toggle {
  -webkit-box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
  box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
}
.btn-group.open .dropdown-toggle.btn-link {
  -webkit-box-shadow: none;
  box-shadow: none;
}
.btn .caret {
  margin-left: 0;
}
.btn-lg .caret {
  border-width: 5px 5px 0;
  border-bottom-width: 0;
}
.dropup .btn-lg .caret {
  border-width: 0 5px 5px;
}
.btn-group-vertical > .btn,
.btn-group-vertical > .btn-group,
.btn-group-vertical > .btn-group > .btn {
  display: block;
  float: none;
  width: 100%;
  max-width: 100%;
}
.btn-group-vertical > .btn-group > .btn {
  float: none;
}
.btn-group-vertical > .btn + .btn,
.btn-group-vertical > .btn + .btn-group,
.btn-group-vertical > .btn-group + .btn,
.btn-group-vertical > .btn-group + .btn-group {
  margin-top: -1px;
  margin-left: 0;
}
.btn-group-vertical > .btn:not(:first-child):not(:last-child) {
  border-radius: 0;
}
.btn-group-vertical > .btn:first-child:not(:last-child) {
  border-top-right-radius: 2px;
  border-top-left-radius: 2px;
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 0;
}
.btn-group-vertical > .btn:last-child:not(:first-child) {
  border-top-right-radius: 0;
  border-top-left-radius: 0;
  border-bottom-right-radius: 2px;
  border-bottom-left-radius: 2px;
}
.btn-group-vertical > .btn-group:not(:first-child):not(:last-child) > .btn {
  border-radius: 0;
}
.btn-group-vertical > .btn-group:first-child:not(:last-child) > .btn:last-child,
.btn-group-vertical > .btn-group:first-child:not(:last-child) > .dropdown-toggle {
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 0;
}
.btn-group-vertical > .btn-group:last-child:not(:first-child) > .btn:first-child {
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.btn-group-justified {
  display: table;
  width: 100%;
  table-layout: fixed;
  border-collapse: separate;
}
.btn-group-justified > .btn,
.btn-group-justified > .btn-group {
  float: none;
  display: table-cell;
  width: 1%;
}
.btn-group-justified > .btn-group .btn {
  width: 100%;
}
.btn-group-justified > .btn-group .dropdown-menu {
  left: auto;
}
[data-toggle="buttons"] > .btn input[type="radio"],
[data-toggle="buttons"] > .btn-group > .btn input[type="radio"],
[data-toggle="buttons"] > .btn input[type="checkbox"],
[data-toggle="buttons"] > .btn-group > .btn input[type="checkbox"] {
  position: absolute;
  clip: rect(0, 0, 0, 0);
  pointer-events: none;
}
.input-group {
  position: relative;
  display: table;
  border-collapse: separate;
}
.input-group[class*="col-"] {
  float: none;
  padding-left: 0;
  padding-right: 0;
}
.input-group .form-control {
  position: relative;
  z-index: 2;
  float: left;
  width: 100%;
  margin-bottom: 0;
}
.input-group .form-control:focus {
  z-index: 3;
}
.input-group-lg > .form-control,
.input-group-lg > .input-group-addon,
.input-group-lg > .input-group-btn > .btn {
  height: 45px;
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
select.input-group-lg > .form-control,
select.input-group-lg > .input-group-addon,
select.input-group-lg > .input-group-btn > .btn {
  height: 45px;
  line-height: 45px;
}
textarea.input-group-lg > .form-control,
textarea.input-group-lg > .input-group-addon,
textarea.input-group-lg > .input-group-btn > .btn,
select[multiple].input-group-lg > .form-control,
select[multiple].input-group-lg > .input-group-addon,
select[multiple].input-group-lg > .input-group-btn > .btn {
  height: auto;
}
.input-group-sm > .form-control,
.input-group-sm > .input-group-addon,
.input-group-sm > .input-group-btn > .btn {
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
select.input-group-sm > .form-control,
select.input-group-sm > .input-group-addon,
select.input-group-sm > .input-group-btn > .btn {
  height: 30px;
  line-height: 30px;
}
textarea.input-group-sm > .form-control,
textarea.input-group-sm > .input-group-addon,
textarea.input-group-sm > .input-group-btn > .btn,
select[multiple].input-group-sm > .form-control,
select[multiple].input-group-sm > .input-group-addon,
select[multiple].input-group-sm > .input-group-btn > .btn {
  height: auto;
}
.input-group-addon,
.input-group-btn,
.input-group .form-control {
  display: table-cell;
}
.input-group-addon:not(:first-child):not(:last-child),
.input-group-btn:not(:first-child):not(:last-child),
.input-group .form-control:not(:first-child):not(:last-child) {
  border-radius: 0;
}
.input-group-addon,
.input-group-btn {
  width: 1%;
  white-space: nowrap;
  vertical-align: middle;
}
.input-group-addon {
  padding: 6px 12px;
  font-size: 13px;
  font-weight: normal;
  line-height: 1;
  color: #555555;
  text-align: center;
  background-color: #eeeeee;
  border: 1px solid #ccc;
  border-radius: 2px;
}
.input-group-addon.input-sm {
  padding: 5px 10px;
  font-size: 12px;
  border-radius: 1px;
}
.input-group-addon.input-lg {
  padding: 10px 16px;
  font-size: 17px;
  border-radius: 3px;
}
.input-group-addon input[type="radio"],
.input-group-addon input[type="checkbox"] {
  margin-top: 0;
}
.input-group .form-control:first-child,
.input-group-addon:first-child,
.input-group-btn:first-child > .btn,
.input-group-btn:first-child > .btn-group > .btn,
.input-group-btn:first-child > .dropdown-toggle,
.input-group-btn:last-child > .btn:not(:last-child):not(.dropdown-toggle),
.input-group-btn:last-child > .btn-group:not(:last-child) > .btn {
  border-bottom-right-radius: 0;
  border-top-right-radius: 0;
}
.input-group-addon:first-child {
  border-right: 0;
}
.input-group .form-control:last-child,
.input-group-addon:last-child,
.input-group-btn:last-child > .btn,
.input-group-btn:last-child > .btn-group > .btn,
.input-group-btn:last-child > .dropdown-toggle,
.input-group-btn:first-child > .btn:not(:first-child),
.input-group-btn:first-child > .btn-group:not(:first-child) > .btn {
  border-bottom-left-radius: 0;
  border-top-left-radius: 0;
}
.input-group-addon:last-child {
  border-left: 0;
}
.input-group-btn {
  position: relative;
  font-size: 0;
  white-space: nowrap;
}
.input-group-btn > .btn {
  position: relative;
}
.input-group-btn > .btn + .btn {
  margin-left: -1px;
}
.input-group-btn > .btn:hover,
.input-group-btn > .btn:focus,
.input-group-btn > .btn:active {
  z-index: 2;
}
.input-group-btn:first-child > .btn,
.input-group-btn:first-child > .btn-group {
  margin-right: -1px;
}
.input-group-btn:last-child > .btn,
.input-group-btn:last-child > .btn-group {
  z-index: 2;
  margin-left: -1px;
}
.nav {
  margin-bottom: 0;
  padding-left: 0;
  list-style: none;
}
.nav > li {
  position: relative;
  display: block;
}
.nav > li > a {
  position: relative;
  display: block;
  padding: 10px 15px;
}
.nav > li > a:hover,
.nav > li > a:focus {
  text-decoration: none;
  background-color: #eeeeee;
}
.nav > li.disabled > a {
  color: #777777;
}
.nav > li.disabled > a:hover,
.nav > li.disabled > a:focus {
  color: #777777;
  text-decoration: none;
  background-color: transparent;
  cursor: not-allowed;
}
.nav .open > a,
.nav .open > a:hover,
.nav .open > a:focus {
  background-color: #eeeeee;
  border-color: #337ab7;
}
.nav .nav-divider {
  height: 1px;
  margin: 8px 0;
  overflow: hidden;
  background-color: #e5e5e5;
}
.nav > li > a > img {
  max-width: none;
}
.nav-tabs {
  border-bottom: 1px solid #ddd;
}
.nav-tabs > li {
  float: left;
  margin-bottom: -1px;
}
.nav-tabs > li > a {
  margin-right: 2px;
  line-height: 1.42857143;
  border: 1px solid transparent;
  border-radius: 2px 2px 0 0;
}
.nav-tabs > li > a:hover {
  border-color: #eeeeee #eeeeee #ddd;
}
.nav-tabs > li.active > a,
.nav-tabs > li.active > a:hover,
.nav-tabs > li.active > a:focus {
  color: #555555;
  background-color: #fff;
  border: 1px solid #ddd;
  border-bottom-color: transparent;
  cursor: default;
}
.nav-tabs.nav-justified {
  width: 100%;
  border-bottom: 0;
}
.nav-tabs.nav-justified > li {
  float: none;
}
.nav-tabs.nav-justified > li > a {
  text-align: center;
  margin-bottom: 5px;
}
.nav-tabs.nav-justified > .dropdown .dropdown-menu {
  top: auto;
  left: auto;
}
@media (min-width: 768px) {
  .nav-tabs.nav-justified > li {
    display: table-cell;
    width: 1%;
  }
  .nav-tabs.nav-justified > li > a {
    margin-bottom: 0;
  }
}
.nav-tabs.nav-justified > li > a {
  margin-right: 0;
  border-radius: 2px;
}
.nav-tabs.nav-justified > .active > a,
.nav-tabs.nav-justified > .active > a:hover,
.nav-tabs.nav-justified > .active > a:focus {
  border: 1px solid #ddd;
}
@media (min-width: 768px) {
  .nav-tabs.nav-justified > li > a {
    border-bottom: 1px solid #ddd;
    border-radius: 2px 2px 0 0;
  }
  .nav-tabs.nav-justified > .active > a,
  .nav-tabs.nav-justified > .active > a:hover,
  .nav-tabs.nav-justified > .active > a:focus {
    border-bottom-color: #fff;
  }
}
.nav-pills > li {
  float: left;
}
.nav-pills > li > a {
  border-radius: 2px;
}
.nav-pills > li + li {
  margin-left: 2px;
}
.nav-pills > li.active > a,
.nav-pills > li.active > a:hover,
.nav-pills > li.active > a:focus {
  color: #fff;
  background-color: #337ab7;
}
.nav-stacked > li {
  float: none;
}
.nav-stacked > li + li {
  margin-top: 2px;
  margin-left: 0;
}
.nav-justified {
  width: 100%;
}
.nav-justified > li {
  float: none;
}
.nav-justified > li > a {
  text-align: center;
  margin-bottom: 5px;
}
.nav-justified > .dropdown .dropdown-menu {
  top: auto;
  left: auto;
}
@media (min-width: 768px) {
  .nav-justified > li {
    display: table-cell;
    width: 1%;
  }
  .nav-justified > li > a {
    margin-bottom: 0;
  }
}
.nav-tabs-justified {
  border-bottom: 0;
}
.nav-tabs-justified > li > a {
  margin-right: 0;
  border-radius: 2px;
}
.nav-tabs-justified > .active > a,
.nav-tabs-justified > .active > a:hover,
.nav-tabs-justified > .active > a:focus {
  border: 1px solid #ddd;
}
@media (min-width: 768px) {
  .nav-tabs-justified > li > a {
    border-bottom: 1px solid #ddd;
    border-radius: 2px 2px 0 0;
  }
  .nav-tabs-justified > .active > a,
  .nav-tabs-justified > .active > a:hover,
  .nav-tabs-justified > .active > a:focus {
    border-bottom-color: #fff;
  }
}
.tab-content > .tab-pane {
  display: none;
}
.tab-content > .active {
  display: block;
}
.nav-tabs .dropdown-menu {
  margin-top: -1px;
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.navbar {
  position: relative;
  min-height: 30px;
  margin-bottom: 18px;
  border: 1px solid transparent;
}
@media (min-width: 541px) {
  .navbar {
    border-radius: 2px;
  }
}
@media (min-width: 541px) {
  .navbar-header {
    float: left;
  }
}
.navbar-collapse {
  overflow-x: visible;
  padding-right: 0px;
  padding-left: 0px;
  border-top: 1px solid transparent;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1);
  -webkit-overflow-scrolling: touch;
}
.navbar-collapse.in {
  overflow-y: auto;
}
@media (min-width: 541px) {
  .navbar-collapse {
    width: auto;
    border-top: 0;
    box-shadow: none;
  }
  .navbar-collapse.collapse {
    display: block !important;
    height: auto !important;
    padding-bottom: 0;
    overflow: visible !important;
  }
  .navbar-collapse.in {
    overflow-y: visible;
  }
  .navbar-fixed-top .navbar-collapse,
  .navbar-static-top .navbar-collapse,
  .navbar-fixed-bottom .navbar-collapse {
    padding-left: 0;
    padding-right: 0;
  }
}
.navbar-fixed-top .navbar-collapse,
.navbar-fixed-bottom .navbar-collapse {
  max-height: 340px;
}
@media (max-device-width: 540px) and (orientation: landscape) {
  .navbar-fixed-top .navbar-collapse,
  .navbar-fixed-bottom .navbar-collapse {
    max-height: 200px;
  }
}
.container > .navbar-header,
.container-fluid > .navbar-header,
.container > .navbar-collapse,
.container-fluid > .navbar-collapse {
  margin-right: 0px;
  margin-left: 0px;
}
@media (min-width: 541px) {
  .container > .navbar-header,
  .container-fluid > .navbar-header,
  .container > .navbar-collapse,
  .container-fluid > .navbar-collapse {
    margin-right: 0;
    margin-left: 0;
  }
}
.navbar-static-top {
  z-index: 1000;
  border-width: 0 0 1px;
}
@media (min-width: 541px) {
  .navbar-static-top {
    border-radius: 0;
  }
}
.navbar-fixed-top,
.navbar-fixed-bottom {
  position: fixed;
  right: 0;
  left: 0;
  z-index: 1030;
}
@media (min-width: 541px) {
  .navbar-fixed-top,
  .navbar-fixed-bottom {
    border-radius: 0;
  }
}
.navbar-fixed-top {
  top: 0;
  border-width: 0 0 1px;
}
.navbar-fixed-bottom {
  bottom: 0;
  margin-bottom: 0;
  border-width: 1px 0 0;
}
.navbar-brand {
  float: left;
  padding: 6px 0px;
  font-size: 17px;
  line-height: 18px;
  height: 30px;
}
.navbar-brand:hover,
.navbar-brand:focus {
  text-decoration: none;
}
.navbar-brand > img {
  display: block;
}
@media (min-width: 541px) {
  .navbar > .container .navbar-brand,
  .navbar > .container-fluid .navbar-brand {
    margin-left: 0px;
  }
}
.navbar-toggle {
  position: relative;
  float: right;
  margin-right: 0px;
  padding: 9px 10px;
  margin-top: -2px;
  margin-bottom: -2px;
  background-color: transparent;
  background-image: none;
  border: 1px solid transparent;
  border-radius: 2px;
}
.navbar-toggle:focus {
  outline: 0;
}
.navbar-toggle .icon-bar {
  display: block;
  width: 22px;
  height: 2px;
  border-radius: 1px;
}
.navbar-toggle .icon-bar + .icon-bar {
  margin-top: 4px;
}
@media (min-width: 541px) {
  .navbar-toggle {
    display: none;
  }
}
.navbar-nav {
  margin: 3px 0px;
}
.navbar-nav > li > a {
  padding-top: 10px;
  padding-bottom: 10px;
  line-height: 18px;
}
@media (max-width: 540px) {
  .navbar-nav .open .dropdown-menu {
    position: static;
    float: none;
    width: auto;
    margin-top: 0;
    background-color: transparent;
    border: 0;
    box-shadow: none;
  }
  .navbar-nav .open .dropdown-menu > li > a,
  .navbar-nav .open .dropdown-menu .dropdown-header {
    padding: 5px 15px 5px 25px;
  }
  .navbar-nav .open .dropdown-menu > li > a {
    line-height: 18px;
  }
  .navbar-nav .open .dropdown-menu > li > a:hover,
  .navbar-nav .open .dropdown-menu > li > a:focus {
    background-image: none;
  }
}
@media (min-width: 541px) {
  .navbar-nav {
    float: left;
    margin: 0;
  }
  .navbar-nav > li {
    float: left;
  }
  .navbar-nav > li > a {
    padding-top: 6px;
    padding-bottom: 6px;
  }
}
.navbar-form {
  margin-left: 0px;
  margin-right: 0px;
  padding: 10px 0px;
  border-top: 1px solid transparent;
  border-bottom: 1px solid transparent;
  -webkit-box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1), 0 1px 0 rgba(255, 255, 255, 0.1);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1), 0 1px 0 rgba(255, 255, 255, 0.1);
  margin-top: -1px;
  margin-bottom: -1px;
}
@media (min-width: 768px) {
  .navbar-form .form-group {
    display: inline-block;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .navbar-form .form-control {
    display: inline-block;
    width: auto;
    vertical-align: middle;
  }
  .navbar-form .form-control-static {
    display: inline-block;
  }
  .navbar-form .input-group {
    display: inline-table;
    vertical-align: middle;
  }
  .navbar-form .input-group .input-group-addon,
  .navbar-form .input-group .input-group-btn,
  .navbar-form .input-group .form-control {
    width: auto;
  }
  .navbar-form .input-group > .form-control {
    width: 100%;
  }
  .navbar-form .control-label {
    margin-bottom: 0;
    vertical-align: middle;
  }
  .navbar-form .radio,
  .navbar-form .checkbox {
    display: inline-block;
    margin-top: 0;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .navbar-form .radio label,
  .navbar-form .checkbox label {
    padding-left: 0;
  }
  .navbar-form .radio input[type="radio"],
  .navbar-form .checkbox input[type="checkbox"] {
    position: relative;
    margin-left: 0;
  }
  .navbar-form .has-feedback .form-control-feedback {
    top: 0;
  }
}
@media (max-width: 540px) {
  .navbar-form .form-group {
    margin-bottom: 5px;
  }
  .navbar-form .form-group:last-child {
    margin-bottom: 0;
  }
}
@media (min-width: 541px) {
  .navbar-form {
    width: auto;
    border: 0;
    margin-left: 0;
    margin-right: 0;
    padding-top: 0;
    padding-bottom: 0;
    -webkit-box-shadow: none;
    box-shadow: none;
  }
}
.navbar-nav > li > .dropdown-menu {
  margin-top: 0;
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.navbar-fixed-bottom .navbar-nav > li > .dropdown-menu {
  margin-bottom: 0;
  border-top-right-radius: 2px;
  border-top-left-radius: 2px;
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 0;
}
.navbar-btn {
  margin-top: -1px;
  margin-bottom: -1px;
}
.navbar-btn.btn-sm {
  margin-top: 0px;
  margin-bottom: 0px;
}
.navbar-btn.btn-xs {
  margin-top: 4px;
  margin-bottom: 4px;
}
.navbar-text {
  margin-top: 6px;
  margin-bottom: 6px;
}
@media (min-width: 541px) {
  .navbar-text {
    float: left;
    margin-left: 0px;
    margin-right: 0px;
  }
}
@media (min-width: 541px) {
  .navbar-left {
    float: left !important;
    float: left;
  }
  .navbar-right {
    float: right !important;
    float: right;
    margin-right: 0px;
  }
  .navbar-right ~ .navbar-right {
    margin-right: 0;
  }
}
.navbar-default {
  background-color: #f8f8f8;
  border-color: #e7e7e7;
}
.navbar-default .navbar-brand {
  color: #777;
}
.navbar-default .navbar-brand:hover,
.navbar-default .navbar-brand:focus {
  color: #5e5e5e;
  background-color: transparent;
}
.navbar-default .navbar-text {
  color: #777;
}
.navbar-default .navbar-nav > li > a {
  color: #777;
}
.navbar-default .navbar-nav > li > a:hover,
.navbar-default .navbar-nav > li > a:focus {
  color: #333;
  background-color: transparent;
}
.navbar-default .navbar-nav > .active > a,
.navbar-default .navbar-nav > .active > a:hover,
.navbar-default .navbar-nav > .active > a:focus {
  color: #555;
  background-color: #e7e7e7;
}
.navbar-default .navbar-nav > .disabled > a,
.navbar-default .navbar-nav > .disabled > a:hover,
.navbar-default .navbar-nav > .disabled > a:focus {
  color: #ccc;
  background-color: transparent;
}
.navbar-default .navbar-toggle {
  border-color: #ddd;
}
.navbar-default .navbar-toggle:hover,
.navbar-default .navbar-toggle:focus {
  background-color: #ddd;
}
.navbar-default .navbar-toggle .icon-bar {
  background-color: #888;
}
.navbar-default .navbar-collapse,
.navbar-default .navbar-form {
  border-color: #e7e7e7;
}
.navbar-default .navbar-nav > .open > a,
.navbar-default .navbar-nav > .open > a:hover,
.navbar-default .navbar-nav > .open > a:focus {
  background-color: #e7e7e7;
  color: #555;
}
@media (max-width: 540px) {
  .navbar-default .navbar-nav .open .dropdown-menu > li > a {
    color: #777;
  }
  .navbar-default .navbar-nav .open .dropdown-menu > li > a:hover,
  .navbar-default .navbar-nav .open .dropdown-menu > li > a:focus {
    color: #333;
    background-color: transparent;
  }
  .navbar-default .navbar-nav .open .dropdown-menu > .active > a,
  .navbar-default .navbar-nav .open .dropdown-menu > .active > a:hover,
  .navbar-default .navbar-nav .open .dropdown-menu > .active > a:focus {
    color: #555;
    background-color: #e7e7e7;
  }
  .navbar-default .navbar-nav .open .dropdown-menu > .disabled > a,
  .navbar-default .navbar-nav .open .dropdown-menu > .disabled > a:hover,
  .navbar-default .navbar-nav .open .dropdown-menu > .disabled > a:focus {
    color: #ccc;
    background-color: transparent;
  }
}
.navbar-default .navbar-link {
  color: #777;
}
.navbar-default .navbar-link:hover {
  color: #333;
}
.navbar-default .btn-link {
  color: #777;
}
.navbar-default .btn-link:hover,
.navbar-default .btn-link:focus {
  color: #333;
}
.navbar-default .btn-link[disabled]:hover,
fieldset[disabled] .navbar-default .btn-link:hover,
.navbar-default .btn-link[disabled]:focus,
fieldset[disabled] .navbar-default .btn-link:focus {
  color: #ccc;
}
.navbar-inverse {
  background-color: #222;
  border-color: #080808;
}
.navbar-inverse .navbar-brand {
  color: #9d9d9d;
}
.navbar-inverse .navbar-brand:hover,
.navbar-inverse .navbar-brand:focus {
  color: #fff;
  background-color: transparent;
}
.navbar-inverse .navbar-text {
  color: #9d9d9d;
}
.navbar-inverse .navbar-nav > li > a {
  color: #9d9d9d;
}
.navbar-inverse .navbar-nav > li > a:hover,
.navbar-inverse .navbar-nav > li > a:focus {
  color: #fff;
  background-color: transparent;
}
.navbar-inverse .navbar-nav > .active > a,
.navbar-inverse .navbar-nav > .active > a:hover,
.navbar-inverse .navbar-nav > .active > a:focus {
  color: #fff;
  background-color: #080808;
}
.navbar-inverse .navbar-nav > .disabled > a,
.navbar-inverse .navbar-nav > .disabled > a:hover,
.navbar-inverse .navbar-nav > .disabled > a:focus {
  color: #444;
  background-color: transparent;
}
.navbar-inverse .navbar-toggle {
  border-color: #333;
}
.navbar-inverse .navbar-toggle:hover,
.navbar-inverse .navbar-toggle:focus {
  background-color: #333;
}
.navbar-inverse .navbar-toggle .icon-bar {
  background-color: #fff;
}
.navbar-inverse .navbar-collapse,
.navbar-inverse .navbar-form {
  border-color: #101010;
}
.navbar-inverse .navbar-nav > .open > a,
.navbar-inverse .navbar-nav > .open > a:hover,
.navbar-inverse .navbar-nav > .open > a:focus {
  background-color: #080808;
  color: #fff;
}
@media (max-width: 540px) {
  .navbar-inverse .navbar-nav .open .dropdown-menu > .dropdown-header {
    border-color: #080808;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu .divider {
    background-color: #080808;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > li > a {
    color: #9d9d9d;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > li > a:hover,
  .navbar-inverse .navbar-nav .open .dropdown-menu > li > a:focus {
    color: #fff;
    background-color: transparent;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > .active > a,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .active > a:hover,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .active > a:focus {
    color: #fff;
    background-color: #080808;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > .disabled > a,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .disabled > a:hover,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .disabled > a:focus {
    color: #444;
    background-color: transparent;
  }
}
.navbar-inverse .navbar-link {
  color: #9d9d9d;
}
.navbar-inverse .navbar-link:hover {
  color: #fff;
}
.navbar-inverse .btn-link {
  color: #9d9d9d;
}
.navbar-inverse .btn-link:hover,
.navbar-inverse .btn-link:focus {
  color: #fff;
}
.navbar-inverse .btn-link[disabled]:hover,
fieldset[disabled] .navbar-inverse .btn-link:hover,
.navbar-inverse .btn-link[disabled]:focus,
fieldset[disabled] .navbar-inverse .btn-link:focus {
  color: #444;
}
.breadcrumb {
  padding: 8px 15px;
  margin-bottom: 18px;
  list-style: none;
  background-color: #f5f5f5;
  border-radius: 2px;
}
.breadcrumb > li {
  display: inline-block;
}
.breadcrumb > li + li:before {
  content: "/\00a0";
  padding: 0 5px;
  color: #5e5e5e;
}
.breadcrumb > .active {
  color: #777777;
}
.pagination {
  display: inline-block;
  padding-left: 0;
  margin: 18px 0;
  border-radius: 2px;
}
.pagination > li {
  display: inline;
}
.pagination > li > a,
.pagination > li > span {
  position: relative;
  float: left;
  padding: 6px 12px;
  line-height: 1.42857143;
  text-decoration: none;
  color: #337ab7;
  background-color: #fff;
  border: 1px solid #ddd;
  margin-left: -1px;
}
.pagination > li:first-child > a,
.pagination > li:first-child > span {
  margin-left: 0;
  border-bottom-left-radius: 2px;
  border-top-left-radius: 2px;
}
.pagination > li:last-child > a,
.pagination > li:last-child > span {
  border-bottom-right-radius: 2px;
  border-top-right-radius: 2px;
}
.pagination > li > a:hover,
.pagination > li > span:hover,
.pagination > li > a:focus,
.pagination > li > span:focus {
  z-index: 2;
  color: #23527c;
  background-color: #eeeeee;
  border-color: #ddd;
}
.pagination > .active > a,
.pagination > .active > span,
.pagination > .active > a:hover,
.pagination > .active > span:hover,
.pagination > .active > a:focus,
.pagination > .active > span:focus {
  z-index: 3;
  color: #fff;
  background-color: #337ab7;
  border-color: #337ab7;
  cursor: default;
}
.pagination > .disabled > span,
.pagination > .disabled > span:hover,
.pagination > .disabled > span:focus,
.pagination > .disabled > a,
.pagination > .disabled > a:hover,
.pagination > .disabled > a:focus {
  color: #777777;
  background-color: #fff;
  border-color: #ddd;
  cursor: not-allowed;
}
.pagination-lg > li > a,
.pagination-lg > li > span {
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
}
.pagination-lg > li:first-child > a,
.pagination-lg > li:first-child > span {
  border-bottom-left-radius: 3px;
  border-top-left-radius: 3px;
}
.pagination-lg > li:last-child > a,
.pagination-lg > li:last-child > span {
  border-bottom-right-radius: 3px;
  border-top-right-radius: 3px;
}
.pagination-sm > li > a,
.pagination-sm > li > span {
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
}
.pagination-sm > li:first-child > a,
.pagination-sm > li:first-child > span {
  border-bottom-left-radius: 1px;
  border-top-left-radius: 1px;
}
.pagination-sm > li:last-child > a,
.pagination-sm > li:last-child > span {
  border-bottom-right-radius: 1px;
  border-top-right-radius: 1px;
}
.pager {
  padding-left: 0;
  margin: 18px 0;
  list-style: none;
  text-align: center;
}
.pager li {
  display: inline;
}
.pager li > a,
.pager li > span {
  display: inline-block;
  padding: 5px 14px;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 15px;
}
.pager li > a:hover,
.pager li > a:focus {
  text-decoration: none;
  background-color: #eeeeee;
}
.pager .next > a,
.pager .next > span {
  float: right;
}
.pager .previous > a,
.pager .previous > span {
  float: left;
}
.pager .disabled > a,
.pager .disabled > a:hover,
.pager .disabled > a:focus,
.pager .disabled > span {
  color: #777777;
  background-color: #fff;
  cursor: not-allowed;
}
.label {
  display: inline;
  padding: .2em .6em .3em;
  font-size: 75%;
  font-weight: bold;
  line-height: 1;
  color: #fff;
  text-align: center;
  white-space: nowrap;
  vertical-align: baseline;
  border-radius: .25em;
}
a.label:hover,
a.label:focus {
  color: #fff;
  text-decoration: none;
  cursor: pointer;
}
.label:empty {
  display: none;
}
.btn .label {
  position: relative;
  top: -1px;
}
.label-default {
  background-color: #777777;
}
.label-default[href]:hover,
.label-default[href]:focus {
  background-color: #5e5e5e;
}
.label-primary {
  background-color: #337ab7;
}
.label-primary[href]:hover,
.label-primary[href]:focus {
  background-color: #286090;
}
.label-success {
  background-color: #5cb85c;
}
.label-success[href]:hover,
.label-success[href]:focus {
  background-color: #449d44;
}
.label-info {
  background-color: #5bc0de;
}
.label-info[href]:hover,
.label-info[href]:focus {
  background-color: #31b0d5;
}
.label-warning {
  background-color: #f0ad4e;
}
.label-warning[href]:hover,
.label-warning[href]:focus {
  background-color: #ec971f;
}
.label-danger {
  background-color: #d9534f;
}
.label-danger[href]:hover,
.label-danger[href]:focus {
  background-color: #c9302c;
}
.badge {
  display: inline-block;
  min-width: 10px;
  padding: 3px 7px;
  font-size: 12px;
  font-weight: bold;
  color: #fff;
  line-height: 1;
  vertical-align: middle;
  white-space: nowrap;
  text-align: center;
  background-color: #777777;
  border-radius: 10px;
}
.badge:empty {
  display: none;
}
.btn .badge {
  position: relative;
  top: -1px;
}
.btn-xs .badge,
.btn-group-xs > .btn .badge {
  top: 0;
  padding: 1px 5px;
}
a.badge:hover,
a.badge:focus {
  color: #fff;
  text-decoration: none;
  cursor: pointer;
}
.list-group-item.active > .badge,
.nav-pills > .active > a > .badge {
  color: #337ab7;
  background-color: #fff;
}
.list-group-item > .badge {
  float: right;
}
.list-group-item > .badge + .badge {
  margin-right: 5px;
}
.nav-pills > li > a > .badge {
  margin-left: 3px;
}
.jumbotron {
  padding-top: 30px;
  padding-bottom: 30px;
  margin-bottom: 30px;
  color: inherit;
  background-color: #eeeeee;
}
.jumbotron h1,
.jumbotron .h1 {
  color: inherit;
}
.jumbotron p {
  margin-bottom: 15px;
  font-size: 20px;
  font-weight: 200;
}
.jumbotron > hr {
  border-top-color: #d5d5d5;
}
.container .jumbotron,
.container-fluid .jumbotron {
  border-radius: 3px;
  padding-left: 0px;
  padding-right: 0px;
}
.jumbotron .container {
  max-width: 100%;
}
@media screen and (min-width: 768px) {
  .jumbotron {
    padding-top: 48px;
    padding-bottom: 48px;
  }
  .container .jumbotron,
  .container-fluid .jumbotron {
    padding-left: 60px;
    padding-right: 60px;
  }
  .jumbotron h1,
  .jumbotron .h1 {
    font-size: 59px;
  }
}
.thumbnail {
  display: block;
  padding: 4px;
  margin-bottom: 18px;
  line-height: 1.42857143;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 2px;
  -webkit-transition: border 0.2s ease-in-out;
  -o-transition: border 0.2s ease-in-out;
  transition: border 0.2s ease-in-out;
}
.thumbnail > img,
.thumbnail a > img {
  margin-left: auto;
  margin-right: auto;
}
a.thumbnail:hover,
a.thumbnail:focus,
a.thumbnail.active {
  border-color: #337ab7;
}
.thumbnail .caption {
  padding: 9px;
  color: #000;
}
.alert {
  padding: 15px;
  margin-bottom: 18px;
  border: 1px solid transparent;
  border-radius: 2px;
}
.alert h4 {
  margin-top: 0;
  color: inherit;
}
.alert .alert-link {
  font-weight: bold;
}
.alert > p,
.alert > ul {
  margin-bottom: 0;
}
.alert > p + p {
  margin-top: 5px;
}
.alert-dismissable,
.alert-dismissible {
  padding-right: 35px;
}
.alert-dismissable .close,
.alert-dismissible .close {
  position: relative;
  top: -2px;
  right: -21px;
  color: inherit;
}
.alert-success {
  background-color: #dff0d8;
  border-color: #d6e9c6;
  color: #3c763d;
}
.alert-success hr {
  border-top-color: #c9e2b3;
}
.alert-success .alert-link {
  color: #2b542c;
}
.alert-info {
  background-color: #d9edf7;
  border-color: #bce8f1;
  color: #31708f;
}
.alert-info hr {
  border-top-color: #a6e1ec;
}
.alert-info .alert-link {
  color: #245269;
}
.alert-warning {
  background-color: #fcf8e3;
  border-color: #faebcc;
  color: #8a6d3b;
}
.alert-warning hr {
  border-top-color: #f7e1b5;
}
.alert-warning .alert-link {
  color: #66512c;
}
.alert-danger {
  background-color: #f2dede;
  border-color: #ebccd1;
  color: #a94442;
}
.alert-danger hr {
  border-top-color: #e4b9c0;
}
.alert-danger .alert-link {
  color: #843534;
}
@-webkit-keyframes progress-bar-stripes {
  from {
    background-position: 40px 0;
  }
  to {
    background-position: 0 0;
  }
}
@keyframes progress-bar-stripes {
  from {
    background-position: 40px 0;
  }
  to {
    background-position: 0 0;
  }
}
.progress {
  overflow: hidden;
  height: 18px;
  margin-bottom: 18px;
  background-color: #f5f5f5;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.1);
  box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.1);
}
.progress-bar {
  float: left;
  width: 0%;
  height: 100%;
  font-size: 12px;
  line-height: 18px;
  color: #fff;
  text-align: center;
  background-color: #337ab7;
  -webkit-box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.15);
  box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.15);
  -webkit-transition: width 0.6s ease;
  -o-transition: width 0.6s ease;
  transition: width 0.6s ease;
}
.progress-striped .progress-bar,
.progress-bar-striped {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-size: 40px 40px;
}
.progress.active .progress-bar,
.progress-bar.active {
  -webkit-animation: progress-bar-stripes 2s linear infinite;
  -o-animation: progress-bar-stripes 2s linear infinite;
  animation: progress-bar-stripes 2s linear infinite;
}
.progress-bar-success {
  background-color: #5cb85c;
}
.progress-striped .progress-bar-success {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.progress-bar-info {
  background-color: #5bc0de;
}
.progress-striped .progress-bar-info {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.progress-bar-warning {
  background-color: #f0ad4e;
}
.progress-striped .progress-bar-warning {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.progress-bar-danger {
  background-color: #d9534f;
}
.progress-striped .progress-bar-danger {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.media {
  margin-top: 15px;
}
.media:first-child {
  margin-top: 0;
}
.media,
.media-body {
  zoom: 1;
  overflow: hidden;
}
.media-body {
  width: 10000px;
}
.media-object {
  display: block;
}
.media-object.img-thumbnail {
  max-width: none;
}
.media-right,
.media > .pull-right {
  padding-left: 10px;
}
.media-left,
.media > .pull-left {
  padding-right: 10px;
}
.media-left,
.media-right,
.media-body {
  display: table-cell;
  vertical-align: top;
}
.media-middle {
  vertical-align: middle;
}
.media-bottom {
  vertical-align: bottom;
}
.media-heading {
  margin-top: 0;
  margin-bottom: 5px;
}
.media-list {
  padding-left: 0;
  list-style: none;
}
.list-group {
  margin-bottom: 20px;
  padding-left: 0;
}
.list-group-item {
  position: relative;
  display: block;
  padding: 10px 15px;
  margin-bottom: -1px;
  background-color: #fff;
  border: 1px solid #ddd;
}
.list-group-item:first-child {
  border-top-right-radius: 2px;
  border-top-left-radius: 2px;
}
.list-group-item:last-child {
  margin-bottom: 0;
  border-bottom-right-radius: 2px;
  border-bottom-left-radius: 2px;
}
a.list-group-item,
button.list-group-item {
  color: #555;
}
a.list-group-item .list-group-item-heading,
button.list-group-item .list-group-item-heading {
  color: #333;
}
a.list-group-item:hover,
button.list-group-item:hover,
a.list-group-item:focus,
button.list-group-item:focus {
  text-decoration: none;
  color: #555;
  background-color: #f5f5f5;
}
button.list-group-item {
  width: 100%;
  text-align: left;
}
.list-group-item.disabled,
.list-group-item.disabled:hover,
.list-group-item.disabled:focus {
  background-color: #eeeeee;
  color: #777777;
  cursor: not-allowed;
}
.list-group-item.disabled .list-group-item-heading,
.list-group-item.disabled:hover .list-group-item-heading,
.list-group-item.disabled:focus .list-group-item-heading {
  color: inherit;
}
.list-group-item.disabled .list-group-item-text,
.list-group-item.disabled:hover .list-group-item-text,
.list-group-item.disabled:focus .list-group-item-text {
  color: #777777;
}
.list-group-item.active,
.list-group-item.active:hover,
.list-group-item.active:focus {
  z-index: 2;
  color: #fff;
  background-color: #337ab7;
  border-color: #337ab7;
}
.list-group-item.active .list-group-item-heading,
.list-group-item.active:hover .list-group-item-heading,
.list-group-item.active:focus .list-group-item-heading,
.list-group-item.active .list-group-item-heading > small,
.list-group-item.active:hover .list-group-item-heading > small,
.list-group-item.active:focus .list-group-item-heading > small,
.list-group-item.active .list-group-item-heading > .small,
.list-group-item.active:hover .list-group-item-heading > .small,
.list-group-item.active:focus .list-group-item-heading > .small {
  color: inherit;
}
.list-group-item.active .list-group-item-text,
.list-group-item.active:hover .list-group-item-text,
.list-group-item.active:focus .list-group-item-text {
  color: #c7ddef;
}
.list-group-item-success {
  color: #3c763d;
  background-color: #dff0d8;
}
a.list-group-item-success,
button.list-group-item-success {
  color: #3c763d;
}
a.list-group-item-success .list-group-item-heading,
button.list-group-item-success .list-group-item-heading {
  color: inherit;
}
a.list-group-item-success:hover,
button.list-group-item-success:hover,
a.list-group-item-success:focus,
button.list-group-item-success:focus {
  color: #3c763d;
  background-color: #d0e9c6;
}
a.list-group-item-success.active,
button.list-group-item-success.active,
a.list-group-item-success.active:hover,
button.list-group-item-success.active:hover,
a.list-group-item-success.active:focus,
button.list-group-item-success.active:focus {
  color: #fff;
  background-color: #3c763d;
  border-color: #3c763d;
}
.list-group-item-info {
  color: #31708f;
  background-color: #d9edf7;
}
a.list-group-item-info,
button.list-group-item-info {
  color: #31708f;
}
a.list-group-item-info .list-group-item-heading,
button.list-group-item-info .list-group-item-heading {
  color: inherit;
}
a.list-group-item-info:hover,
button.list-group-item-info:hover,
a.list-group-item-info:focus,
button.list-group-item-info:focus {
  color: #31708f;
  background-color: #c4e3f3;
}
a.list-group-item-info.active,
button.list-group-item-info.active,
a.list-group-item-info.active:hover,
button.list-group-item-info.active:hover,
a.list-group-item-info.active:focus,
button.list-group-item-info.active:focus {
  color: #fff;
  background-color: #31708f;
  border-color: #31708f;
}
.list-group-item-warning {
  color: #8a6d3b;
  background-color: #fcf8e3;
}
a.list-group-item-warning,
button.list-group-item-warning {
  color: #8a6d3b;
}
a.list-group-item-warning .list-group-item-heading,
button.list-group-item-warning .list-group-item-heading {
  color: inherit;
}
a.list-group-item-warning:hover,
button.list-group-item-warning:hover,
a.list-group-item-warning:focus,
button.list-group-item-warning:focus {
  color: #8a6d3b;
  background-color: #faf2cc;
}
a.list-group-item-warning.active,
button.list-group-item-warning.active,
a.list-group-item-warning.active:hover,
button.list-group-item-warning.active:hover,
a.list-group-item-warning.active:focus,
button.list-group-item-warning.active:focus {
  color: #fff;
  background-color: #8a6d3b;
  border-color: #8a6d3b;
}
.list-group-item-danger {
  color: #a94442;
  background-color: #f2dede;
}
a.list-group-item-danger,
button.list-group-item-danger {
  color: #a94442;
}
a.list-group-item-danger .list-group-item-heading,
button.list-group-item-danger .list-group-item-heading {
  color: inherit;
}
a.list-group-item-danger:hover,
button.list-group-item-danger:hover,
a.list-group-item-danger:focus,
button.list-group-item-danger:focus {
  color: #a94442;
  background-color: #ebcccc;
}
a.list-group-item-danger.active,
button.list-group-item-danger.active,
a.list-group-item-danger.active:hover,
button.list-group-item-danger.active:hover,
a.list-group-item-danger.active:focus,
button.list-group-item-danger.active:focus {
  color: #fff;
  background-color: #a94442;
  border-color: #a94442;
}
.list-group-item-heading {
  margin-top: 0;
  margin-bottom: 5px;
}
.list-group-item-text {
  margin-bottom: 0;
  line-height: 1.3;
}
.panel {
  margin-bottom: 18px;
  background-color: #fff;
  border: 1px solid transparent;
  border-radius: 2px;
  -webkit-box-shadow: 0 1px 1px rgba(0, 0, 0, 0.05);
  box-shadow: 0 1px 1px rgba(0, 0, 0, 0.05);
}
.panel-body {
  padding: 15px;
}
.panel-heading {
  padding: 10px 15px;
  border-bottom: 1px solid transparent;
  border-top-right-radius: 1px;
  border-top-left-radius: 1px;
}
.panel-heading > .dropdown .dropdown-toggle {
  color: inherit;
}
.panel-title {
  margin-top: 0;
  margin-bottom: 0;
  font-size: 15px;
  color: inherit;
}
.panel-title > a,
.panel-title > small,
.panel-title > .small,
.panel-title > small > a,
.panel-title > .small > a {
  color: inherit;
}
.panel-footer {
  padding: 10px 15px;
  background-color: #f5f5f5;
  border-top: 1px solid #ddd;
  border-bottom-right-radius: 1px;
  border-bottom-left-radius: 1px;
}
.panel > .list-group,
.panel > .panel-collapse > .list-group {
  margin-bottom: 0;
}
.panel > .list-group .list-group-item,
.panel > .panel-collapse > .list-group .list-group-item {
  border-width: 1px 0;
  border-radius: 0;
}
.panel > .list-group:first-child .list-group-item:first-child,
.panel > .panel-collapse > .list-group:first-child .list-group-item:first-child {
  border-top: 0;
  border-top-right-radius: 1px;
  border-top-left-radius: 1px;
}
.panel > .list-group:last-child .list-group-item:last-child,
.panel > .panel-collapse > .list-group:last-child .list-group-item:last-child {
  border-bottom: 0;
  border-bottom-right-radius: 1px;
  border-bottom-left-radius: 1px;
}
.panel > .panel-heading + .panel-collapse > .list-group .list-group-item:first-child {
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.panel-heading + .list-group .list-group-item:first-child {
  border-top-width: 0;
}
.list-group + .panel-footer {
  border-top-width: 0;
}
.panel > .table,
.panel > .table-responsive > .table,
.panel > .panel-collapse > .table {
  margin-bottom: 0;
}
.panel > .table caption,
.panel > .table-responsive > .table caption,
.panel > .panel-collapse > .table caption {
  padding-left: 15px;
  padding-right: 15px;
}
.panel > .table:first-child,
.panel > .table-responsive:first-child > .table:first-child {
  border-top-right-radius: 1px;
  border-top-left-radius: 1px;
}
.panel > .table:first-child > thead:first-child > tr:first-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child,
.panel > .table:first-child > tbody:first-child > tr:first-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child {
  border-top-left-radius: 1px;
  border-top-right-radius: 1px;
}
.panel > .table:first-child > thead:first-child > tr:first-child td:first-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child td:first-child,
.panel > .table:first-child > tbody:first-child > tr:first-child td:first-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child td:first-child,
.panel > .table:first-child > thead:first-child > tr:first-child th:first-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child th:first-child,
.panel > .table:first-child > tbody:first-child > tr:first-child th:first-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child th:first-child {
  border-top-left-radius: 1px;
}
.panel > .table:first-child > thead:first-child > tr:first-child td:last-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child td:last-child,
.panel > .table:first-child > tbody:first-child > tr:first-child td:last-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child td:last-child,
.panel > .table:first-child > thead:first-child > tr:first-child th:last-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child th:last-child,
.panel > .table:first-child > tbody:first-child > tr:first-child th:last-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child th:last-child {
  border-top-right-radius: 1px;
}
.panel > .table:last-child,
.panel > .table-responsive:last-child > .table:last-child {
  border-bottom-right-radius: 1px;
  border-bottom-left-radius: 1px;
}
.panel > .table:last-child > tbody:last-child > tr:last-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child {
  border-bottom-left-radius: 1px;
  border-bottom-right-radius: 1px;
}
.panel > .table:last-child > tbody:last-child > tr:last-child td:first-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child td:first-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child td:first-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child td:first-child,
.panel > .table:last-child > tbody:last-child > tr:last-child th:first-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child th:first-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child th:first-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child th:first-child {
  border-bottom-left-radius: 1px;
}
.panel > .table:last-child > tbody:last-child > tr:last-child td:last-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child td:last-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child td:last-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child td:last-child,
.panel > .table:last-child > tbody:last-child > tr:last-child th:last-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child th:last-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child th:last-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child th:last-child {
  border-bottom-right-radius: 1px;
}
.panel > .panel-body + .table,
.panel > .panel-body + .table-responsive,
.panel > .table + .panel-body,
.panel > .table-responsive + .panel-body {
  border-top: 1px solid #ddd;
}
.panel > .table > tbody:first-child > tr:first-child th,
.panel > .table > tbody:first-child > tr:first-child td {
  border-top: 0;
}
.panel > .table-bordered,
.panel > .table-responsive > .table-bordered {
  border: 0;
}
.panel > .table-bordered > thead > tr > th:first-child,
.panel > .table-responsive > .table-bordered > thead > tr > th:first-child,
.panel > .table-bordered > tbody > tr > th:first-child,
.panel > .table-responsive > .table-bordered > tbody > tr > th:first-child,
.panel > .table-bordered > tfoot > tr > th:first-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > th:first-child,
.panel > .table-bordered > thead > tr > td:first-child,
.panel > .table-responsive > .table-bordered > thead > tr > td:first-child,
.panel > .table-bordered > tbody > tr > td:first-child,
.panel > .table-responsive > .table-bordered > tbody > tr > td:first-child,
.panel > .table-bordered > tfoot > tr > td:first-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > td:first-child {
  border-left: 0;
}
.panel > .table-bordered > thead > tr > th:last-child,
.panel > .table-responsive > .table-bordered > thead > tr > th:last-child,
.panel > .table-bordered > tbody > tr > th:last-child,
.panel > .table-responsive > .table-bordered > tbody > tr > th:last-child,
.panel > .table-bordered > tfoot > tr > th:last-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > th:last-child,
.panel > .table-bordered > thead > tr > td:last-child,
.panel > .table-responsive > .table-bordered > thead > tr > td:last-child,
.panel > .table-bordered > tbody > tr > td:last-child,
.panel > .table-responsive > .table-bordered > tbody > tr > td:last-child,
.panel > .table-bordered > tfoot > tr > td:last-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > td:last-child {
  border-right: 0;
}
.panel > .table-bordered > thead > tr:first-child > td,
.panel > .table-responsive > .table-bordered > thead > tr:first-child > td,
.panel > .table-bordered > tbody > tr:first-child > td,
.panel > .table-responsive > .table-bordered > tbody > tr:first-child > td,
.panel > .table-bordered > thead > tr:first-child > th,
.panel > .table-responsive > .table-bordered > thead > tr:first-child > th,
.panel > .table-bordered > tbody > tr:first-child > th,
.panel > .table-responsive > .table-bordered > tbody > tr:first-child > th {
  border-bottom: 0;
}
.panel > .table-bordered > tbody > tr:last-child > td,
.panel > .table-responsive > .table-bordered > tbody > tr:last-child > td,
.panel > .table-bordered > tfoot > tr:last-child > td,
.panel > .table-responsive > .table-bordered > tfoot > tr:last-child > td,
.panel > .table-bordered > tbody > tr:last-child > th,
.panel > .table-responsive > .table-bordered > tbody > tr:last-child > th,
.panel > .table-bordered > tfoot > tr:last-child > th,
.panel > .table-responsive > .table-bordered > tfoot > tr:last-child > th {
  border-bottom: 0;
}
.panel > .table-responsive {
  border: 0;
  margin-bottom: 0;
}
.panel-group {
  margin-bottom: 18px;
}
.panel-group .panel {
  margin-bottom: 0;
  border-radius: 2px;
}
.panel-group .panel + .panel {
  margin-top: 5px;
}
.panel-group .panel-heading {
  border-bottom: 0;
}
.panel-group .panel-heading + .panel-collapse > .panel-body,
.panel-group .panel-heading + .panel-collapse > .list-group {
  border-top: 1px solid #ddd;
}
.panel-group .panel-footer {
  border-top: 0;
}
.panel-group .panel-footer + .panel-collapse .panel-body {
  border-bottom: 1px solid #ddd;
}
.panel-default {
  border-color: #ddd;
}
.panel-default > .panel-heading {
  color: #333333;
  background-color: #f5f5f5;
  border-color: #ddd;
}
.panel-default > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #ddd;
}
.panel-default > .panel-heading .badge {
  color: #f5f5f5;
  background-color: #333333;
}
.panel-default > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #ddd;
}
.panel-primary {
  border-color: #337ab7;
}
.panel-primary > .panel-heading {
  color: #fff;
  background-color: #337ab7;
  border-color: #337ab7;
}
.panel-primary > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #337ab7;
}
.panel-primary > .panel-heading .badge {
  color: #337ab7;
  background-color: #fff;
}
.panel-primary > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #337ab7;
}
.panel-success {
  border-color: #d6e9c6;
}
.panel-success > .panel-heading {
  color: #3c763d;
  background-color: #dff0d8;
  border-color: #d6e9c6;
}
.panel-success > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #d6e9c6;
}
.panel-success > .panel-heading .badge {
  color: #dff0d8;
  background-color: #3c763d;
}
.panel-success > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #d6e9c6;
}
.panel-info {
  border-color: #bce8f1;
}
.panel-info > .panel-heading {
  color: #31708f;
  background-color: #d9edf7;
  border-color: #bce8f1;
}
.panel-info > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #bce8f1;
}
.panel-info > .panel-heading .badge {
  color: #d9edf7;
  background-color: #31708f;
}
.panel-info > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #bce8f1;
}
.panel-warning {
  border-color: #faebcc;
}
.panel-warning > .panel-heading {
  color: #8a6d3b;
  background-color: #fcf8e3;
  border-color: #faebcc;
}
.panel-warning > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #faebcc;
}
.panel-warning > .panel-heading .badge {
  color: #fcf8e3;
  background-color: #8a6d3b;
}
.panel-warning > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #faebcc;
}
.panel-danger {
  border-color: #ebccd1;
}
.panel-danger > .panel-heading {
  color: #a94442;
  background-color: #f2dede;
  border-color: #ebccd1;
}
.panel-danger > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #ebccd1;
}
.panel-danger > .panel-heading .badge {
  color: #f2dede;
  background-color: #a94442;
}
.panel-danger > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #ebccd1;
}
.embed-responsive {
  position: relative;
  display: block;
  height: 0;
  padding: 0;
  overflow: hidden;
}
.embed-responsive .embed-responsive-item,
.embed-responsive iframe,
.embed-responsive embed,
.embed-responsive object,
.embed-responsive video {
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  height: 100%;
  width: 100%;
  border: 0;
}
.embed-responsive-16by9 {
  padding-bottom: 56.25%;
}
.embed-responsive-4by3 {
  padding-bottom: 75%;
}
.well {
  min-height: 20px;
  padding: 19px;
  margin-bottom: 20px;
  background-color: #f5f5f5;
  border: 1px solid #e3e3e3;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.05);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.05);
}
.well blockquote {
  border-color: #ddd;
  border-color: rgba(0, 0, 0, 0.15);
}
.well-lg {
  padding: 24px;
  border-radius: 3px;
}
.well-sm {
  padding: 9px;
  border-radius: 1px;
}
.close {
  float: right;
  font-size: 19.5px;
  font-weight: bold;
  line-height: 1;
  color: #000;
  text-shadow: 0 1px 0 #fff;
  opacity: 0.2;
  filter: alpha(opacity=20);
}
.close:hover,
.close:focus {
  color: #000;
  text-decoration: none;
  cursor: pointer;
  opacity: 0.5;
  filter: alpha(opacity=50);
}
button.close {
  padding: 0;
  cursor: pointer;
  background: transparent;
  border: 0;
  -webkit-appearance: none;
}
.modal-open {
  overflow: hidden;
}
.modal {
  display: none;
  overflow: hidden;
  position: fixed;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;
  z-index: 1050;
  -webkit-overflow-scrolling: touch;
  outline: 0;
}
.modal.fade .modal-dialog {
  -webkit-transform: translate(0, -25%);
  -ms-transform: translate(0, -25%);
  -o-transform: translate(0, -25%);
  transform: translate(0, -25%);
  -webkit-transition: -webkit-transform 0.3s ease-out;
  -moz-transition: -moz-transform 0.3s ease-out;
  -o-transition: -o-transform 0.3s ease-out;
  transition: transform 0.3s ease-out;
}
.modal.in .modal-dialog {
  -webkit-transform: translate(0, 0);
  -ms-transform: translate(0, 0);
  -o-transform: translate(0, 0);
  transform: translate(0, 0);
}
.modal-open .modal {
  overflow-x: hidden;
  overflow-y: auto;
}
.modal-dialog {
  position: relative;
  width: auto;
  margin: 10px;
}
.modal-content {
  position: relative;
  background-color: #fff;
  border: 1px solid #999;
  border: 1px solid rgba(0, 0, 0, 0.2);
  border-radius: 3px;
  -webkit-box-shadow: 0 3px 9px rgba(0, 0, 0, 0.5);
  box-shadow: 0 3px 9px rgba(0, 0, 0, 0.5);
  background-clip: padding-box;
  outline: 0;
}
.modal-backdrop {
  position: fixed;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;
  z-index: 1040;
  background-color: #000;
}
.modal-backdrop.fade {
  opacity: 0;
  filter: alpha(opacity=0);
}
.modal-backdrop.in {
  opacity: 0.5;
  filter: alpha(opacity=50);
}
.modal-header {
  padding: 15px;
  border-bottom: 1px solid #e5e5e5;
}
.modal-header .close {
  margin-top: -2px;
}
.modal-title {
  margin: 0;
  line-height: 1.42857143;
}
.modal-body {
  position: relative;
  padding: 15px;
}
.modal-footer {
  padding: 15px;
  text-align: right;
  border-top: 1px solid #e5e5e5;
}
.modal-footer .btn + .btn {
  margin-left: 5px;
  margin-bottom: 0;
}
.modal-footer .btn-group .btn + .btn {
  margin-left: -1px;
}
.modal-footer .btn-block + .btn-block {
  margin-left: 0;
}
.modal-scrollbar-measure {
  position: absolute;
  top: -9999px;
  width: 50px;
  height: 50px;
  overflow: scroll;
}
@media (min-width: 768px) {
  .modal-dialog {
    width: 600px;
    margin: 30px auto;
  }
  .modal-content {
    -webkit-box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
  }
  .modal-sm {
    width: 300px;
  }
}
@media (min-width: 992px) {
  .modal-lg {
    width: 900px;
  }
}
.tooltip {
  position: absolute;
  z-index: 1070;
  display: block;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-style: normal;
  font-weight: normal;
  letter-spacing: normal;
  line-break: auto;
  line-height: 1.42857143;
  text-align: left;
  text-align: start;
  text-decoration: none;
  text-shadow: none;
  text-transform: none;
  white-space: normal;
  word-break: normal;
  word-spacing: normal;
  word-wrap: normal;
  font-size: 12px;
  opacity: 0;
  filter: alpha(opacity=0);
}
.tooltip.in {
  opacity: 0.9;
  filter: alpha(opacity=90);
}
.tooltip.top {
  margin-top: -3px;
  padding: 5px 0;
}
.tooltip.right {
  margin-left: 3px;
  padding: 0 5px;
}
.tooltip.bottom {
  margin-top: 3px;
  padding: 5px 0;
}
.tooltip.left {
  margin-left: -3px;
  padding: 0 5px;
}
.tooltip-inner {
  max-width: 200px;
  padding: 3px 8px;
  color: #fff;
  text-align: center;
  background-color: #000;
  border-radius: 2px;
}
.tooltip-arrow {
  position: absolute;
  width: 0;
  height: 0;
  border-color: transparent;
  border-style: solid;
}
.tooltip.top .tooltip-arrow {
  bottom: 0;
  left: 50%;
  margin-left: -5px;
  border-width: 5px 5px 0;
  border-top-color: #000;
}
.tooltip.top-left .tooltip-arrow {
  bottom: 0;
  right: 5px;
  margin-bottom: -5px;
  border-width: 5px 5px 0;
  border-top-color: #000;
}
.tooltip.top-right .tooltip-arrow {
  bottom: 0;
  left: 5px;
  margin-bottom: -5px;
  border-width: 5px 5px 0;
  border-top-color: #000;
}
.tooltip.right .tooltip-arrow {
  top: 50%;
  left: 0;
  margin-top: -5px;
  border-width: 5px 5px 5px 0;
  border-right-color: #000;
}
.tooltip.left .tooltip-arrow {
  top: 50%;
  right: 0;
  margin-top: -5px;
  border-width: 5px 0 5px 5px;
  border-left-color: #000;
}
.tooltip.bottom .tooltip-arrow {
  top: 0;
  left: 50%;
  margin-left: -5px;
  border-width: 0 5px 5px;
  border-bottom-color: #000;
}
.tooltip.bottom-left .tooltip-arrow {
  top: 0;
  right: 5px;
  margin-top: -5px;
  border-width: 0 5px 5px;
  border-bottom-color: #000;
}
.tooltip.bottom-right .tooltip-arrow {
  top: 0;
  left: 5px;
  margin-top: -5px;
  border-width: 0 5px 5px;
  border-bottom-color: #000;
}
.popover {
  position: absolute;
  top: 0;
  left: 0;
  z-index: 1060;
  display: none;
  max-width: 276px;
  padding: 1px;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-style: normal;
  font-weight: normal;
  letter-spacing: normal;
  line-break: auto;
  line-height: 1.42857143;
  text-align: left;
  text-align: start;
  text-decoration: none;
  text-shadow: none;
  text-transform: none;
  white-space: normal;
  word-break: normal;
  word-spacing: normal;
  word-wrap: normal;
  font-size: 13px;
  background-color: #fff;
  background-clip: padding-box;
  border: 1px solid #ccc;
  border: 1px solid rgba(0, 0, 0, 0.2);
  border-radius: 3px;
  -webkit-box-shadow: 0 5px 10px rgba(0, 0, 0, 0.2);
  box-shadow: 0 5px 10px rgba(0, 0, 0, 0.2);
}
.popover.top {
  margin-top: -10px;
}
.popover.right {
  margin-left: 10px;
}
.popover.bottom {
  margin-top: 10px;
}
.popover.left {
  margin-left: -10px;
}
.popover-title {
  margin: 0;
  padding: 8px 14px;
  font-size: 13px;
  background-color: #f7f7f7;
  border-bottom: 1px solid #ebebeb;
  border-radius: 2px 2px 0 0;
}
.popover-content {
  padding: 9px 14px;
}
.popover > .arrow,
.popover > .arrow:after {
  position: absolute;
  display: block;
  width: 0;
  height: 0;
  border-color: transparent;
  border-style: solid;
}
.popover > .arrow {
  border-width: 11px;
}
.popover > .arrow:after {
  border-width: 10px;
  content: "";
}
.popover.top > .arrow {
  left: 50%;
  margin-left: -11px;
  border-bottom-width: 0;
  border-top-color: #999999;
  border-top-color: rgba(0, 0, 0, 0.25);
  bottom: -11px;
}
.popover.top > .arrow:after {
  content: " ";
  bottom: 1px;
  margin-left: -10px;
  border-bottom-width: 0;
  border-top-color: #fff;
}
.popover.right > .arrow {
  top: 50%;
  left: -11px;
  margin-top: -11px;
  border-left-width: 0;
  border-right-color: #999999;
  border-right-color: rgba(0, 0, 0, 0.25);
}
.popover.right > .arrow:after {
  content: " ";
  left: 1px;
  bottom: -10px;
  border-left-width: 0;
  border-right-color: #fff;
}
.popover.bottom > .arrow {
  left: 50%;
  margin-left: -11px;
  border-top-width: 0;
  border-bottom-color: #999999;
  border-bottom-color: rgba(0, 0, 0, 0.25);
  top: -11px;
}
.popover.bottom > .arrow:after {
  content: " ";
  top: 1px;
  margin-left: -10px;
  border-top-width: 0;
  border-bottom-color: #fff;
}
.popover.left > .arrow {
  top: 50%;
  right: -11px;
  margin-top: -11px;
  border-right-width: 0;
  border-left-color: #999999;
  border-left-color: rgba(0, 0, 0, 0.25);
}
.popover.left > .arrow:after {
  content: " ";
  right: 1px;
  border-right-width: 0;
  border-left-color: #fff;
  bottom: -10px;
}
.carousel {
  position: relative;
}
.carousel-inner {
  position: relative;
  overflow: hidden;
  width: 100%;
}
.carousel-inner > .item {
  display: none;
  position: relative;
  -webkit-transition: 0.6s ease-in-out left;
  -o-transition: 0.6s ease-in-out left;
  transition: 0.6s ease-in-out left;
}
.carousel-inner > .item > img,
.carousel-inner > .item > a > img {
  line-height: 1;
}
@media all and (transform-3d), (-webkit-transform-3d) {
  .carousel-inner > .item {
    -webkit-transition: -webkit-transform 0.6s ease-in-out;
    -moz-transition: -moz-transform 0.6s ease-in-out;
    -o-transition: -o-transform 0.6s ease-in-out;
    transition: transform 0.6s ease-in-out;
    -webkit-backface-visibility: hidden;
    -moz-backface-visibility: hidden;
    backface-visibility: hidden;
    -webkit-perspective: 1000px;
    -moz-perspective: 1000px;
    perspective: 1000px;
  }
  .carousel-inner > .item.next,
  .carousel-inner > .item.active.right {
    -webkit-transform: translate3d(100%, 0, 0);
    transform: translate3d(100%, 0, 0);
    left: 0;
  }
  .carousel-inner > .item.prev,
  .carousel-inner > .item.active.left {
    -webkit-transform: translate3d(-100%, 0, 0);
    transform: translate3d(-100%, 0, 0);
    left: 0;
  }
  .carousel-inner > .item.next.left,
  .carousel-inner > .item.prev.right,
  .carousel-inner > .item.active {
    -webkit-transform: translate3d(0, 0, 0);
    transform: translate3d(0, 0, 0);
    left: 0;
  }
}
.carousel-inner > .active,
.carousel-inner > .next,
.carousel-inner > .prev {
  display: block;
}
.carousel-inner > .active {
  left: 0;
}
.carousel-inner > .next,
.carousel-inner > .prev {
  position: absolute;
  top: 0;
  width: 100%;
}
.carousel-inner > .next {
  left: 100%;
}
.carousel-inner > .prev {
  left: -100%;
}
.carousel-inner > .next.left,
.carousel-inner > .prev.right {
  left: 0;
}
.carousel-inner > .active.left {
  left: -100%;
}
.carousel-inner > .active.right {
  left: 100%;
}
.carousel-control {
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  width: 15%;
  opacity: 0.5;
  filter: alpha(opacity=50);
  font-size: 20px;
  color: #fff;
  text-align: center;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6);
  background-color: rgba(0, 0, 0, 0);
}
.carousel-control.left {
  background-image: -webkit-linear-gradient(left, rgba(0, 0, 0, 0.5) 0%, rgba(0, 0, 0, 0.0001) 100%);
  background-image: -o-linear-gradient(left, rgba(0, 0, 0, 0.5) 0%, rgba(0, 0, 0, 0.0001) 100%);
  background-image: linear-gradient(to right, rgba(0, 0, 0, 0.5) 0%, rgba(0, 0, 0, 0.0001) 100%);
  background-repeat: repeat-x;
  filter: progid:DXImageTransform.Microsoft.gradient(startColorstr='#80000000', endColorstr='#00000000', GradientType=1);
}
.carousel-control.right {
  left: auto;
  right: 0;
  background-image: -webkit-linear-gradient(left, rgba(0, 0, 0, 0.0001) 0%, rgba(0, 0, 0, 0.5) 100%);
  background-image: -o-linear-gradient(left, rgba(0, 0, 0, 0.0001) 0%, rgba(0, 0, 0, 0.5) 100%);
  background-image: linear-gradient(to right, rgba(0, 0, 0, 0.0001) 0%, rgba(0, 0, 0, 0.5) 100%);
  background-repeat: repeat-x;
  filter: progid:DXImageTransform.Microsoft.gradient(startColorstr='#00000000', endColorstr='#80000000', GradientType=1);
}
.carousel-control:hover,
.carousel-control:focus {
  outline: 0;
  color: #fff;
  text-decoration: none;
  opacity: 0.9;
  filter: alpha(opacity=90);
}
.carousel-control .icon-prev,
.carousel-control .icon-next,
.carousel-control .glyphicon-chevron-left,
.carousel-control .glyphicon-chevron-right {
  position: absolute;
  top: 50%;
  margin-top: -10px;
  z-index: 5;
  display: inline-block;
}
.carousel-control .icon-prev,
.carousel-control .glyphicon-chevron-left {
  left: 50%;
  margin-left: -10px;
}
.carousel-control .icon-next,
.carousel-control .glyphicon-chevron-right {
  right: 50%;
  margin-right: -10px;
}
.carousel-control .icon-prev,
.carousel-control .icon-next {
  width: 20px;
  height: 20px;
  line-height: 1;
  font-family: serif;
}
.carousel-control .icon-prev:before {
  content: '\2039';
}
.carousel-control .icon-next:before {
  content: '\203a';
}
.carousel-indicators {
  position: absolute;
  bottom: 10px;
  left: 50%;
  z-index: 15;
  width: 60%;
  margin-left: -30%;
  padding-left: 0;
  list-style: none;
  text-align: center;
}
.carousel-indicators li {
  display: inline-block;
  width: 10px;
  height: 10px;
  margin: 1px;
  text-indent: -999px;
  border: 1px solid #fff;
  border-radius: 10px;
  cursor: pointer;
  background-color: #000 \9;
  background-color: rgba(0, 0, 0, 0);
}
.carousel-indicators .active {
  margin: 0;
  width: 12px;
  height: 12px;
  background-color: #fff;
}
.carousel-caption {
  position: absolute;
  left: 15%;
  right: 15%;
  bottom: 20px;
  z-index: 10;
  padding-top: 20px;
  padding-bottom: 20px;
  color: #fff;
  text-align: center;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6);
}
.carousel-caption .btn {
  text-shadow: none;
}
@media screen and (min-width: 768px) {
  .carousel-control .glyphicon-chevron-left,
  .carousel-control .glyphicon-chevron-right,
  .carousel-control .icon-prev,
  .carousel-control .icon-next {
    width: 30px;
    height: 30px;
    margin-top: -10px;
    font-size: 30px;
  }
  .carousel-control .glyphicon-chevron-left,
  .carousel-control .icon-prev {
    margin-left: -10px;
  }
  .carousel-control .glyphicon-chevron-right,
  .carousel-control .icon-next {
    margin-right: -10px;
  }
  .carousel-caption {
    left: 20%;
    right: 20%;
    padding-bottom: 30px;
  }
  .carousel-indicators {
    bottom: 20px;
  }
}
.clearfix:before,
.clearfix:after,
.dl-horizontal dd:before,
.dl-horizontal dd:after,
.container:before,
.container:after,
.container-fluid:before,
.container-fluid:after,
.row:before,
.row:after,
.form-horizontal .form-group:before,
.form-horizontal .form-group:after,
.btn-toolbar:before,
.btn-toolbar:after,
.btn-group-vertical > .btn-group:before,
.btn-group-vertical > .btn-group:after,
.nav:before,
.nav:after,
.navbar:before,
.navbar:after,
.navbar-header:before,
.navbar-header:after,
.navbar-collapse:before,
.navbar-collapse:after,
.pager:before,
.pager:after,
.panel-body:before,
.panel-body:after,
.modal-header:before,
.modal-header:after,
.modal-footer:before,
.modal-footer:after,
.item_buttons:before,
.item_buttons:after {
  content: " ";
  display: table;
}
.clearfix:after,
.dl-horizontal dd:after,
.container:after,
.container-fluid:after,
.row:after,
.form-horizontal .form-group:after,
.btn-toolbar:after,
.btn-group-vertical > .btn-group:after,
.nav:after,
.navbar:after,
.navbar-header:after,
.navbar-collapse:after,
.pager:after,
.panel-body:after,
.modal-header:after,
.modal-footer:after,
.item_buttons:after {
  clear: both;
}
.center-block {
  display: block;
  margin-left: auto;
  margin-right: auto;
}
.pull-right {
  float: right !important;
}
.pull-left {
  float: left !important;
}
.hide {
  display: none !important;
}
.show {
  display: block !important;
}
.invisible {
  visibility: hidden;
}
.text-hide {
  font: 0/0 a;
  color: transparent;
  text-shadow: none;
  background-color: transparent;
  border: 0;
}
.hidden {
  display: none !important;
}
.affix {
  position: fixed;
}
@-ms-viewport {
  width: device-width;
}
.visible-xs,
.visible-sm,
.visible-md,
.visible-lg {
  display: none !important;
}
.visible-xs-block,
.visible-xs-inline,
.visible-xs-inline-block,
.visible-sm-block,
.visible-sm-inline,
.visible-sm-inline-block,
.visible-md-block,
.visible-md-inline,
.visible-md-inline-block,
.visible-lg-block,
.visible-lg-inline,
.visible-lg-inline-block {
  display: none !important;
}
@media (max-width: 767px) {
  .visible-xs {
    display: block !important;
  }
  table.visible-xs {
    display: table !important;
  }
  tr.visible-xs {
    display: table-row !important;
  }
  th.visible-xs,
  td.visible-xs {
    display: table-cell !important;
  }
}
@media (max-width: 767px) {
  .visible-xs-block {
    display: block !important;
  }
}
@media (max-width: 767px) {
  .visible-xs-inline {
    display: inline !important;
  }
}
@media (max-width: 767px) {
  .visible-xs-inline-block {
    display: inline-block !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm {
    display: block !important;
  }
  table.visible-sm {
    display: table !important;
  }
  tr.visible-sm {
    display: table-row !important;
  }
  th.visible-sm,
  td.visible-sm {
    display: table-cell !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm-block {
    display: block !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm-inline {
    display: inline !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm-inline-block {
    display: inline-block !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md {
    display: block !important;
  }
  table.visible-md {
    display: table !important;
  }
  tr.visible-md {
    display: table-row !important;
  }
  th.visible-md,
  td.visible-md {
    display: table-cell !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md-block {
    display: block !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md-inline {
    display: inline !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md-inline-block {
    display: inline-block !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg {
    display: block !important;
  }
  table.visible-lg {
    display: table !important;
  }
  tr.visible-lg {
    display: table-row !important;
  }
  th.visible-lg,
  td.visible-lg {
    display: table-cell !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg-block {
    display: block !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg-inline {
    display: inline !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg-inline-block {
    display: inline-block !important;
  }
}
@media (max-width: 767px) {
  .hidden-xs {
    display: none !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .hidden-sm {
    display: none !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .hidden-md {
    display: none !important;
  }
}
@media (min-width: 1200px) {
  .hidden-lg {
    display: none !important;
  }
}
.visible-print {
  display: none !important;
}
@media print {
  .visible-print {
    display: block !important;
  }
  table.visible-print {
    display: table !important;
  }
  tr.visible-print {
    display: table-row !important;
  }
  th.visible-print,
  td.visible-print {
    display: table-cell !important;
  }
}
.visible-print-block {
  display: none !important;
}
@media print {
  .visible-print-block {
    display: block !important;
  }
}
.visible-print-inline {
  display: none !important;
}
@media print {
  .visible-print-inline {
    display: inline !important;
  }
}
.visible-print-inline-block {
  display: none !important;
}
@media print {
  .visible-print-inline-block {
    display: inline-block !important;
  }
}
@media print {
  .hidden-print {
    display: none !important;
  }
}
/*!
*
* Font Awesome
*
*/
/*!
 *  Font Awesome 4.7.0 by @davegandy - http://fontawesome.io - @fontawesome
 *  License - http://fontawesome.io/license (Font: SIL OFL 1.1, CSS: MIT License)
 */
/* FONT PATH
 * -------------------------- */
@font-face {
  font-family: 'FontAwesome';
  src: url('../components/font-awesome/fonts/fontawesome-webfont.eot?v=4.7.0');
  src: url('../components/font-awesome/fonts/fontawesome-webfont.eot?#iefix&v=4.7.0') format('embedded-opentype'), url('../components/font-awesome/fonts/fontawesome-webfont.woff2?v=4.7.0') format('woff2'), url('../components/font-awesome/fonts/fontawesome-webfont.woff?v=4.7.0') format('woff'), url('../components/font-awesome/fonts/fontawesome-webfont.ttf?v=4.7.0') format('truetype'), url('../components/font-awesome/fonts/fontawesome-webfont.svg?v=4.7.0#fontawesomeregular') format('svg');
  font-weight: normal;
  font-style: normal;
}
.fa {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
/* makes the font 33% larger relative to the icon container */
.fa-lg {
  font-size: 1.33333333em;
  line-height: 0.75em;
  vertical-align: -15%;
}
.fa-2x {
  font-size: 2em;
}
.fa-3x {
  font-size: 3em;
}
.fa-4x {
  font-size: 4em;
}
.fa-5x {
  font-size: 5em;
}
.fa-fw {
  width: 1.28571429em;
  text-align: center;
}
.fa-ul {
  padding-left: 0;
  margin-left: 2.14285714em;
  list-style-type: none;
}
.fa-ul > li {
  position: relative;
}
.fa-li {
  position: absolute;
  left: -2.14285714em;
  width: 2.14285714em;
  top: 0.14285714em;
  text-align: center;
}
.fa-li.fa-lg {
  left: -1.85714286em;
}
.fa-border {
  padding: .2em .25em .15em;
  border: solid 0.08em #eee;
  border-radius: .1em;
}
.fa-pull-left {
  float: left;
}
.fa-pull-right {
  float: right;
}
.fa.fa-pull-left {
  margin-right: .3em;
}
.fa.fa-pull-right {
  margin-left: .3em;
}
/* Deprecated as of 4.4.0 */
.pull-right {
  float: right;
}
.pull-left {
  float: left;
}
.fa.pull-left {
  margin-right: .3em;
}
.fa.pull-right {
  margin-left: .3em;
}
.fa-spin {
  -webkit-animation: fa-spin 2s infinite linear;
  animation: fa-spin 2s infinite linear;
}
.fa-pulse {
  -webkit-animation: fa-spin 1s infinite steps(8);
  animation: fa-spin 1s infinite steps(8);
}
@-webkit-keyframes fa-spin {
  0% {
    -webkit-transform: rotate(0deg);
    transform: rotate(0deg);
  }
  100% {
    -webkit-transform: rotate(359deg);
    transform: rotate(359deg);
  }
}
@keyframes fa-spin {
  0% {
    -webkit-transform: rotate(0deg);
    transform: rotate(0deg);
  }
  100% {
    -webkit-transform: rotate(359deg);
    transform: rotate(359deg);
  }
}
.fa-rotate-90 {
  -ms-filter: "progid:DXImageTransform.Microsoft.BasicImage(rotation=1)";
  -webkit-transform: rotate(90deg);
  -ms-transform: rotate(90deg);
  transform: rotate(90deg);
}
.fa-rotate-180 {
  -ms-filter: "progid:DXImageTransform.Microsoft.BasicImage(rotation=2)";
  -webkit-transform: rotate(180deg);
  -ms-transform: rotate(180deg);
  transform: rotate(180deg);
}
.fa-rotate-270 {
  -ms-filter: "progid:DXImageTransform.Microsoft.BasicImage(rotation=3)";
  -webkit-transform: rotate(270deg);
  -ms-transform: rotate(270deg);
  transform: rotate(270deg);
}
.fa-flip-horizontal {
  -ms-filter: "progid:DXImageTransform.Microsoft.BasicImage(rotation=0, mirror=1)";
  -webkit-transform: scale(-1, 1);
  -ms-transform: scale(-1, 1);
  transform: scale(-1, 1);
}
.fa-flip-vertical {
  -ms-filter: "progid:DXImageTransform.Microsoft.BasicImage(rotation=2, mirror=1)";
  -webkit-transform: scale(1, -1);
  -ms-transform: scale(1, -1);
  transform: scale(1, -1);
}
:root .fa-rotate-90,
:root .fa-rotate-180,
:root .fa-rotate-270,
:root .fa-flip-horizontal,
:root .fa-flip-vertical {
  filter: none;
}
.fa-stack {
  position: relative;
  display: inline-block;
  width: 2em;
  height: 2em;
  line-height: 2em;
  vertical-align: middle;
}
.fa-stack-1x,
.fa-stack-2x {
  position: absolute;
  left: 0;
  width: 100%;
  text-align: center;
}
.fa-stack-1x {
  line-height: inherit;
}
.fa-stack-2x {
  font-size: 2em;
}
.fa-inverse {
  color: #fff;
}
/* Font Awesome uses the Unicode Private Use Area (PUA) to ensure screen
   readers do not read off random characters that represent icons */
.fa-glass:before {
  content: "\f000";
}
.fa-music:before {
  content: "\f001";
}
.fa-search:before {
  content: "\f002";
}
.fa-envelope-o:before {
  content: "\f003";
}
.fa-heart:before {
  content: "\f004";
}
.fa-star:before {
  content: "\f005";
}
.fa-star-o:before {
  content: "\f006";
}
.fa-user:before {
  content: "\f007";
}
.fa-film:before {
  content: "\f008";
}
.fa-th-large:before {
  content: "\f009";
}
.fa-th:before {
  content: "\f00a";
}
.fa-th-list:before {
  content: "\f00b";
}
.fa-check:before {
  content: "\f00c";
}
.fa-remove:before,
.fa-close:before,
.fa-times:before {
  content: "\f00d";
}
.fa-search-plus:before {
  content: "\f00e";
}
.fa-search-minus:before {
  content: "\f010";
}
.fa-power-off:before {
  content: "\f011";
}
.fa-signal:before {
  content: "\f012";
}
.fa-gear:before,
.fa-cog:before {
  content: "\f013";
}
.fa-trash-o:before {
  content: "\f014";
}
.fa-home:before {
  content: "\f015";
}
.fa-file-o:before {
  content: "\f016";
}
.fa-clock-o:before {
  content: "\f017";
}
.fa-road:before {
  content: "\f018";
}
.fa-download:before {
  content: "\f019";
}
.fa-arrow-circle-o-down:before {
  content: "\f01a";
}
.fa-arrow-circle-o-up:before {
  content: "\f01b";
}
.fa-inbox:before {
  content: "\f01c";
}
.fa-play-circle-o:before {
  content: "\f01d";
}
.fa-rotate-right:before,
.fa-repeat:before {
  content: "\f01e";
}
.fa-refresh:before {
  content: "\f021";
}
.fa-list-alt:before {
  content: "\f022";
}
.fa-lock:before {
  content: "\f023";
}
.fa-flag:before {
  content: "\f024";
}
.fa-headphones:before {
  content: "\f025";
}
.fa-volume-off:before {
  content: "\f026";
}
.fa-volume-down:before {
  content: "\f027";
}
.fa-volume-up:before {
  content: "\f028";
}
.fa-qrcode:before {
  content: "\f029";
}
.fa-barcode:before {
  content: "\f02a";
}
.fa-tag:before {
  content: "\f02b";
}
.fa-tags:before {
  content: "\f02c";
}
.fa-book:before {
  content: "\f02d";
}
.fa-bookmark:before {
  content: "\f02e";
}
.fa-print:before {
  content: "\f02f";
}
.fa-camera:before {
  content: "\f030";
}
.fa-font:before {
  content: "\f031";
}
.fa-bold:before {
  content: "\f032";
}
.fa-italic:before {
  content: "\f033";
}
.fa-text-height:before {
  content: "\f034";
}
.fa-text-width:before {
  content: "\f035";
}
.fa-align-left:before {
  content: "\f036";
}
.fa-align-center:before {
  content: "\f037";
}
.fa-align-right:before {
  content: "\f038";
}
.fa-align-justify:before {
  content: "\f039";
}
.fa-list:before {
  content: "\f03a";
}
.fa-dedent:before,
.fa-outdent:before {
  content: "\f03b";
}
.fa-indent:before {
  content: "\f03c";
}
.fa-video-camera:before {
  content: "\f03d";
}
.fa-photo:before,
.fa-image:before,
.fa-picture-o:before {
  content: "\f03e";
}
.fa-pencil:before {
  content: "\f040";
}
.fa-map-marker:before {
  content: "\f041";
}
.fa-adjust:before {
  content: "\f042";
}
.fa-tint:before {
  content: "\f043";
}
.fa-edit:before,
.fa-pencil-square-o:before {
  content: "\f044";
}
.fa-share-square-o:before {
  content: "\f045";
}
.fa-check-square-o:before {
  content: "\f046";
}
.fa-arrows:before {
  content: "\f047";
}
.fa-step-backward:before {
  content: "\f048";
}
.fa-fast-backward:before {
  content: "\f049";
}
.fa-backward:before {
  content: "\f04a";
}
.fa-play:before {
  content: "\f04b";
}
.fa-pause:before {
  content: "\f04c";
}
.fa-stop:before {
  content: "\f04d";
}
.fa-forward:before {
  content: "\f04e";
}
.fa-fast-forward:before {
  content: "\f050";
}
.fa-step-forward:before {
  content: "\f051";
}
.fa-eject:before {
  content: "\f052";
}
.fa-chevron-left:before {
  content: "\f053";
}
.fa-chevron-right:before {
  content: "\f054";
}
.fa-plus-circle:before {
  content: "\f055";
}
.fa-minus-circle:before {
  content: "\f056";
}
.fa-times-circle:before {
  content: "\f057";
}
.fa-check-circle:before {
  content: "\f058";
}
.fa-question-circle:before {
  content: "\f059";
}
.fa-info-circle:before {
  content: "\f05a";
}
.fa-crosshairs:before {
  content: "\f05b";
}
.fa-times-circle-o:before {
  content: "\f05c";
}
.fa-check-circle-o:before {
  content: "\f05d";
}
.fa-ban:before {
  content: "\f05e";
}
.fa-arrow-left:before {
  content: "\f060";
}
.fa-arrow-right:before {
  content: "\f061";
}
.fa-arrow-up:before {
  content: "\f062";
}
.fa-arrow-down:before {
  content: "\f063";
}
.fa-mail-forward:before,
.fa-share:before {
  content: "\f064";
}
.fa-expand:before {
  content: "\f065";
}
.fa-compress:before {
  content: "\f066";
}
.fa-plus:before {
  content: "\f067";
}
.fa-minus:before {
  content: "\f068";
}
.fa-asterisk:before {
  content: "\f069";
}
.fa-exclamation-circle:before {
  content: "\f06a";
}
.fa-gift:before {
  content: "\f06b";
}
.fa-leaf:before {
  content: "\f06c";
}
.fa-fire:before {
  content: "\f06d";
}
.fa-eye:before {
  content: "\f06e";
}
.fa-eye-slash:before {
  content: "\f070";
}
.fa-warning:before,
.fa-exclamation-triangle:before {
  content: "\f071";
}
.fa-plane:before {
  content: "\f072";
}
.fa-calendar:before {
  content: "\f073";
}
.fa-random:before {
  content: "\f074";
}
.fa-comment:before {
  content: "\f075";
}
.fa-magnet:before {
  content: "\f076";
}
.fa-chevron-up:before {
  content: "\f077";
}
.fa-chevron-down:before {
  content: "\f078";
}
.fa-retweet:before {
  content: "\f079";
}
.fa-shopping-cart:before {
  content: "\f07a";
}
.fa-folder:before {
  content: "\f07b";
}
.fa-folder-open:before {
  content: "\f07c";
}
.fa-arrows-v:before {
  content: "\f07d";
}
.fa-arrows-h:before {
  content: "\f07e";
}
.fa-bar-chart-o:before,
.fa-bar-chart:before {
  content: "\f080";
}
.fa-twitter-square:before {
  content: "\f081";
}
.fa-facebook-square:before {
  content: "\f082";
}
.fa-camera-retro:before {
  content: "\f083";
}
.fa-key:before {
  content: "\f084";
}
.fa-gears:before,
.fa-cogs:before {
  content: "\f085";
}
.fa-comments:before {
  content: "\f086";
}
.fa-thumbs-o-up:before {
  content: "\f087";
}
.fa-thumbs-o-down:before {
  content: "\f088";
}
.fa-star-half:before {
  content: "\f089";
}
.fa-heart-o:before {
  content: "\f08a";
}
.fa-sign-out:before {
  content: "\f08b";
}
.fa-linkedin-square:before {
  content: "\f08c";
}
.fa-thumb-tack:before {
  content: "\f08d";
}
.fa-external-link:before {
  content: "\f08e";
}
.fa-sign-in:before {
  content: "\f090";
}
.fa-trophy:before {
  content: "\f091";
}
.fa-github-square:before {
  content: "\f092";
}
.fa-upload:before {
  content: "\f093";
}
.fa-lemon-o:before {
  content: "\f094";
}
.fa-phone:before {
  content: "\f095";
}
.fa-square-o:before {
  content: "\f096";
}
.fa-bookmark-o:before {
  content: "\f097";
}
.fa-phone-square:before {
  content: "\f098";
}
.fa-twitter:before {
  content: "\f099";
}
.fa-facebook-f:before,
.fa-facebook:before {
  content: "\f09a";
}
.fa-github:before {
  content: "\f09b";
}
.fa-unlock:before {
  content: "\f09c";
}
.fa-credit-card:before {
  content: "\f09d";
}
.fa-feed:before,
.fa-rss:before {
  content: "\f09e";
}
.fa-hdd-o:before {
  content: "\f0a0";
}
.fa-bullhorn:before {
  content: "\f0a1";
}
.fa-bell:before {
  content: "\f0f3";
}
.fa-certificate:before {
  content: "\f0a3";
}
.fa-hand-o-right:before {
  content: "\f0a4";
}
.fa-hand-o-left:before {
  content: "\f0a5";
}
.fa-hand-o-up:before {
  content: "\f0a6";
}
.fa-hand-o-down:before {
  content: "\f0a7";
}
.fa-arrow-circle-left:before {
  content: "\f0a8";
}
.fa-arrow-circle-right:before {
  content: "\f0a9";
}
.fa-arrow-circle-up:before {
  content: "\f0aa";
}
.fa-arrow-circle-down:before {
  content: "\f0ab";
}
.fa-globe:before {
  content: "\f0ac";
}
.fa-wrench:before {
  content: "\f0ad";
}
.fa-tasks:before {
  content: "\f0ae";
}
.fa-filter:before {
  content: "\f0b0";
}
.fa-briefcase:before {
  content: "\f0b1";
}
.fa-arrows-alt:before {
  content: "\f0b2";
}
.fa-group:before,
.fa-users:before {
  content: "\f0c0";
}
.fa-chain:before,
.fa-link:before {
  content: "\f0c1";
}
.fa-cloud:before {
  content: "\f0c2";
}
.fa-flask:before {
  content: "\f0c3";
}
.fa-cut:before,
.fa-scissors:before {
  content: "\f0c4";
}
.fa-copy:before,
.fa-files-o:before {
  content: "\f0c5";
}
.fa-paperclip:before {
  content: "\f0c6";
}
.fa-save:before,
.fa-floppy-o:before {
  content: "\f0c7";
}
.fa-square:before {
  content: "\f0c8";
}
.fa-navicon:before,
.fa-reorder:before,
.fa-bars:before {
  content: "\f0c9";
}
.fa-list-ul:before {
  content: "\f0ca";
}
.fa-list-ol:before {
  content: "\f0cb";
}
.fa-strikethrough:before {
  content: "\f0cc";
}
.fa-underline:before {
  content: "\f0cd";
}
.fa-table:before {
  content: "\f0ce";
}
.fa-magic:before {
  content: "\f0d0";
}
.fa-truck:before {
  content: "\f0d1";
}
.fa-pinterest:before {
  content: "\f0d2";
}
.fa-pinterest-square:before {
  content: "\f0d3";
}
.fa-google-plus-square:before {
  content: "\f0d4";
}
.fa-google-plus:before {
  content: "\f0d5";
}
.fa-money:before {
  content: "\f0d6";
}
.fa-caret-down:before {
  content: "\f0d7";
}
.fa-caret-up:before {
  content: "\f0d8";
}
.fa-caret-left:before {
  content: "\f0d9";
}
.fa-caret-right:before {
  content: "\f0da";
}
.fa-columns:before {
  content: "\f0db";
}
.fa-unsorted:before,
.fa-sort:before {
  content: "\f0dc";
}
.fa-sort-down:before,
.fa-sort-desc:before {
  content: "\f0dd";
}
.fa-sort-up:before,
.fa-sort-asc:before {
  content: "\f0de";
}
.fa-envelope:before {
  content: "\f0e0";
}
.fa-linkedin:before {
  content: "\f0e1";
}
.fa-rotate-left:before,
.fa-undo:before {
  content: "\f0e2";
}
.fa-legal:before,
.fa-gavel:before {
  content: "\f0e3";
}
.fa-dashboard:before,
.fa-tachometer:before {
  content: "\f0e4";
}
.fa-comment-o:before {
  content: "\f0e5";
}
.fa-comments-o:before {
  content: "\f0e6";
}
.fa-flash:before,
.fa-bolt:before {
  content: "\f0e7";
}
.fa-sitemap:before {
  content: "\f0e8";
}
.fa-umbrella:before {
  content: "\f0e9";
}
.fa-paste:before,
.fa-clipboard:before {
  content: "\f0ea";
}
.fa-lightbulb-o:before {
  content: "\f0eb";
}
.fa-exchange:before {
  content: "\f0ec";
}
.fa-cloud-download:before {
  content: "\f0ed";
}
.fa-cloud-upload:before {
  content: "\f0ee";
}
.fa-user-md:before {
  content: "\f0f0";
}
.fa-stethoscope:before {
  content: "\f0f1";
}
.fa-suitcase:before {
  content: "\f0f2";
}
.fa-bell-o:before {
  content: "\f0a2";
}
.fa-coffee:before {
  content: "\f0f4";
}
.fa-cutlery:before {
  content: "\f0f5";
}
.fa-file-text-o:before {
  content: "\f0f6";
}
.fa-building-o:before {
  content: "\f0f7";
}
.fa-hospital-o:before {
  content: "\f0f8";
}
.fa-ambulance:before {
  content: "\f0f9";
}
.fa-medkit:before {
  content: "\f0fa";
}
.fa-fighter-jet:before {
  content: "\f0fb";
}
.fa-beer:before {
  content: "\f0fc";
}
.fa-h-square:before {
  content: "\f0fd";
}
.fa-plus-square:before {
  content: "\f0fe";
}
.fa-angle-double-left:before {
  content: "\f100";
}
.fa-angle-double-right:before {
  content: "\f101";
}
.fa-angle-double-up:before {
  content: "\f102";
}
.fa-angle-double-down:before {
  content: "\f103";
}
.fa-angle-left:before {
  content: "\f104";
}
.fa-angle-right:before {
  content: "\f105";
}
.fa-angle-up:before {
  content: "\f106";
}
.fa-angle-down:before {
  content: "\f107";
}
.fa-desktop:before {
  content: "\f108";
}
.fa-laptop:before {
  content: "\f109";
}
.fa-tablet:before {
  content: "\f10a";
}
.fa-mobile-phone:before,
.fa-mobile:before {
  content: "\f10b";
}
.fa-circle-o:before {
  content: "\f10c";
}
.fa-quote-left:before {
  content: "\f10d";
}
.fa-quote-right:before {
  content: "\f10e";
}
.fa-spinner:before {
  content: "\f110";
}
.fa-circle:before {
  content: "\f111";
}
.fa-mail-reply:before,
.fa-reply:before {
  content: "\f112";
}
.fa-github-alt:before {
  content: "\f113";
}
.fa-folder-o:before {
  content: "\f114";
}
.fa-folder-open-o:before {
  content: "\f115";
}
.fa-smile-o:before {
  content: "\f118";
}
.fa-frown-o:before {
  content: "\f119";
}
.fa-meh-o:before {
  content: "\f11a";
}
.fa-gamepad:before {
  content: "\f11b";
}
.fa-keyboard-o:before {
  content: "\f11c";
}
.fa-flag-o:before {
  content: "\f11d";
}
.fa-flag-checkered:before {
  content: "\f11e";
}
.fa-terminal:before {
  content: "\f120";
}
.fa-code:before {
  content: "\f121";
}
.fa-mail-reply-all:before,
.fa-reply-all:before {
  content: "\f122";
}
.fa-star-half-empty:before,
.fa-star-half-full:before,
.fa-star-half-o:before {
  content: "\f123";
}
.fa-location-arrow:before {
  content: "\f124";
}
.fa-crop:before {
  content: "\f125";
}
.fa-code-fork:before {
  content: "\f126";
}
.fa-unlink:before,
.fa-chain-broken:before {
  content: "\f127";
}
.fa-question:before {
  content: "\f128";
}
.fa-info:before {
  content: "\f129";
}
.fa-exclamation:before {
  content: "\f12a";
}
.fa-superscript:before {
  content: "\f12b";
}
.fa-subscript:before {
  content: "\f12c";
}
.fa-eraser:before {
  content: "\f12d";
}
.fa-puzzle-piece:before {
  content: "\f12e";
}
.fa-microphone:before {
  content: "\f130";
}
.fa-microphone-slash:before {
  content: "\f131";
}
.fa-shield:before {
  content: "\f132";
}
.fa-calendar-o:before {
  content: "\f133";
}
.fa-fire-extinguisher:before {
  content: "\f134";
}
.fa-rocket:before {
  content: "\f135";
}
.fa-maxcdn:before {
  content: "\f136";
}
.fa-chevron-circle-left:before {
  content: "\f137";
}
.fa-chevron-circle-right:before {
  content: "\f138";
}
.fa-chevron-circle-up:before {
  content: "\f139";
}
.fa-chevron-circle-down:before {
  content: "\f13a";
}
.fa-html5:before {
  content: "\f13b";
}
.fa-css3:before {
  content: "\f13c";
}
.fa-anchor:before {
  content: "\f13d";
}
.fa-unlock-alt:before {
  content: "\f13e";
}
.fa-bullseye:before {
  content: "\f140";
}
.fa-ellipsis-h:before {
  content: "\f141";
}
.fa-ellipsis-v:before {
  content: "\f142";
}
.fa-rss-square:before {
  content: "\f143";
}
.fa-play-circle:before {
  content: "\f144";
}
.fa-ticket:before {
  content: "\f145";
}
.fa-minus-square:before {
  content: "\f146";
}
.fa-minus-square-o:before {
  content: "\f147";
}
.fa-level-up:before {
  content: "\f148";
}
.fa-level-down:before {
  content: "\f149";
}
.fa-check-square:before {
  content: "\f14a";
}
.fa-pencil-square:before {
  content: "\f14b";
}
.fa-external-link-square:before {
  content: "\f14c";
}
.fa-share-square:before {
  content: "\f14d";
}
.fa-compass:before {
  content: "\f14e";
}
.fa-toggle-down:before,
.fa-caret-square-o-down:before {
  content: "\f150";
}
.fa-toggle-up:before,
.fa-caret-square-o-up:before {
  content: "\f151";
}
.fa-toggle-right:before,
.fa-caret-square-o-right:before {
  content: "\f152";
}
.fa-euro:before,
.fa-eur:before {
  content: "\f153";
}
.fa-gbp:before {
  content: "\f154";
}
.fa-dollar:before,
.fa-usd:before {
  content: "\f155";
}
.fa-rupee:before,
.fa-inr:before {
  content: "\f156";
}
.fa-cny:before,
.fa-rmb:before,
.fa-yen:before,
.fa-jpy:before {
  content: "\f157";
}
.fa-ruble:before,
.fa-rouble:before,
.fa-rub:before {
  content: "\f158";
}
.fa-won:before,
.fa-krw:before {
  content: "\f159";
}
.fa-bitcoin:before,
.fa-btc:before {
  content: "\f15a";
}
.fa-file:before {
  content: "\f15b";
}
.fa-file-text:before {
  content: "\f15c";
}
.fa-sort-alpha-asc:before {
  content: "\f15d";
}
.fa-sort-alpha-desc:before {
  content: "\f15e";
}
.fa-sort-amount-asc:before {
  content: "\f160";
}
.fa-sort-amount-desc:before {
  content: "\f161";
}
.fa-sort-numeric-asc:before {
  content: "\f162";
}
.fa-sort-numeric-desc:before {
  content: "\f163";
}
.fa-thumbs-up:before {
  content: "\f164";
}
.fa-thumbs-down:before {
  content: "\f165";
}
.fa-youtube-square:before {
  content: "\f166";
}
.fa-youtube:before {
  content: "\f167";
}
.fa-xing:before {
  content: "\f168";
}
.fa-xing-square:before {
  content: "\f169";
}
.fa-youtube-play:before {
  content: "\f16a";
}
.fa-dropbox:before {
  content: "\f16b";
}
.fa-stack-overflow:before {
  content: "\f16c";
}
.fa-instagram:before {
  content: "\f16d";
}
.fa-flickr:before {
  content: "\f16e";
}
.fa-adn:before {
  content: "\f170";
}
.fa-bitbucket:before {
  content: "\f171";
}
.fa-bitbucket-square:before {
  content: "\f172";
}
.fa-tumblr:before {
  content: "\f173";
}
.fa-tumblr-square:before {
  content: "\f174";
}
.fa-long-arrow-down:before {
  content: "\f175";
}
.fa-long-arrow-up:before {
  content: "\f176";
}
.fa-long-arrow-left:before {
  content: "\f177";
}
.fa-long-arrow-right:before {
  content: "\f178";
}
.fa-apple:before {
  content: "\f179";
}
.fa-windows:before {
  content: "\f17a";
}
.fa-android:before {
  content: "\f17b";
}
.fa-linux:before {
  content: "\f17c";
}
.fa-dribbble:before {
  content: "\f17d";
}
.fa-skype:before {
  content: "\f17e";
}
.fa-foursquare:before {
  content: "\f180";
}
.fa-trello:before {
  content: "\f181";
}
.fa-female:before {
  content: "\f182";
}
.fa-male:before {
  content: "\f183";
}
.fa-gittip:before,
.fa-gratipay:before {
  content: "\f184";
}
.fa-sun-o:before {
  content: "\f185";
}
.fa-moon-o:before {
  content: "\f186";
}
.fa-archive:before {
  content: "\f187";
}
.fa-bug:before {
  content: "\f188";
}
.fa-vk:before {
  content: "\f189";
}
.fa-weibo:before {
  content: "\f18a";
}
.fa-renren:before {
  content: "\f18b";
}
.fa-pagelines:before {
  content: "\f18c";
}
.fa-stack-exchange:before {
  content: "\f18d";
}
.fa-arrow-circle-o-right:before {
  content: "\f18e";
}
.fa-arrow-circle-o-left:before {
  content: "\f190";
}
.fa-toggle-left:before,
.fa-caret-square-o-left:before {
  content: "\f191";
}
.fa-dot-circle-o:before {
  content: "\f192";
}
.fa-wheelchair:before {
  content: "\f193";
}
.fa-vimeo-square:before {
  content: "\f194";
}
.fa-turkish-lira:before,
.fa-try:before {
  content: "\f195";
}
.fa-plus-square-o:before {
  content: "\f196";
}
.fa-space-shuttle:before {
  content: "\f197";
}
.fa-slack:before {
  content: "\f198";
}
.fa-envelope-square:before {
  content: "\f199";
}
.fa-wordpress:before {
  content: "\f19a";
}
.fa-openid:before {
  content: "\f19b";
}
.fa-institution:before,
.fa-bank:before,
.fa-university:before {
  content: "\f19c";
}
.fa-mortar-board:before,
.fa-graduation-cap:before {
  content: "\f19d";
}
.fa-yahoo:before {
  content: "\f19e";
}
.fa-google:before {
  content: "\f1a0";
}
.fa-reddit:before {
  content: "\f1a1";
}
.fa-reddit-square:before {
  content: "\f1a2";
}
.fa-stumbleupon-circle:before {
  content: "\f1a3";
}
.fa-stumbleupon:before {
  content: "\f1a4";
}
.fa-delicious:before {
  content: "\f1a5";
}
.fa-digg:before {
  content: "\f1a6";
}
.fa-pied-piper-pp:before {
  content: "\f1a7";
}
.fa-pied-piper-alt:before {
  content: "\f1a8";
}
.fa-drupal:before {
  content: "\f1a9";
}
.fa-joomla:before {
  content: "\f1aa";
}
.fa-language:before {
  content: "\f1ab";
}
.fa-fax:before {
  content: "\f1ac";
}
.fa-building:before {
  content: "\f1ad";
}
.fa-child:before {
  content: "\f1ae";
}
.fa-paw:before {
  content: "\f1b0";
}
.fa-spoon:before {
  content: "\f1b1";
}
.fa-cube:before {
  content: "\f1b2";
}
.fa-cubes:before {
  content: "\f1b3";
}
.fa-behance:before {
  content: "\f1b4";
}
.fa-behance-square:before {
  content: "\f1b5";
}
.fa-steam:before {
  content: "\f1b6";
}
.fa-steam-square:before {
  content: "\f1b7";
}
.fa-recycle:before {
  content: "\f1b8";
}
.fa-automobile:before,
.fa-car:before {
  content: "\f1b9";
}
.fa-cab:before,
.fa-taxi:before {
  content: "\f1ba";
}
.fa-tree:before {
  content: "\f1bb";
}
.fa-spotify:before {
  content: "\f1bc";
}
.fa-deviantart:before {
  content: "\f1bd";
}
.fa-soundcloud:before {
  content: "\f1be";
}
.fa-database:before {
  content: "\f1c0";
}
.fa-file-pdf-o:before {
  content: "\f1c1";
}
.fa-file-word-o:before {
  content: "\f1c2";
}
.fa-file-excel-o:before {
  content: "\f1c3";
}
.fa-file-powerpoint-o:before {
  content: "\f1c4";
}
.fa-file-photo-o:before,
.fa-file-picture-o:before,
.fa-file-image-o:before {
  content: "\f1c5";
}
.fa-file-zip-o:before,
.fa-file-archive-o:before {
  content: "\f1c6";
}
.fa-file-sound-o:before,
.fa-file-audio-o:before {
  content: "\f1c7";
}
.fa-file-movie-o:before,
.fa-file-video-o:before {
  content: "\f1c8";
}
.fa-file-code-o:before {
  content: "\f1c9";
}
.fa-vine:before {
  content: "\f1ca";
}
.fa-codepen:before {
  content: "\f1cb";
}
.fa-jsfiddle:before {
  content: "\f1cc";
}
.fa-life-bouy:before,
.fa-life-buoy:before,
.fa-life-saver:before,
.fa-support:before,
.fa-life-ring:before {
  content: "\f1cd";
}
.fa-circle-o-notch:before {
  content: "\f1ce";
}
.fa-ra:before,
.fa-resistance:before,
.fa-rebel:before {
  content: "\f1d0";
}
.fa-ge:before,
.fa-empire:before {
  content: "\f1d1";
}
.fa-git-square:before {
  content: "\f1d2";
}
.fa-git:before {
  content: "\f1d3";
}
.fa-y-combinator-square:before,
.fa-yc-square:before,
.fa-hacker-news:before {
  content: "\f1d4";
}
.fa-tencent-weibo:before {
  content: "\f1d5";
}
.fa-qq:before {
  content: "\f1d6";
}
.fa-wechat:before,
.fa-weixin:before {
  content: "\f1d7";
}
.fa-send:before,
.fa-paper-plane:before {
  content: "\f1d8";
}
.fa-send-o:before,
.fa-paper-plane-o:before {
  content: "\f1d9";
}
.fa-history:before {
  content: "\f1da";
}
.fa-circle-thin:before {
  content: "\f1db";
}
.fa-header:before {
  content: "\f1dc";
}
.fa-paragraph:before {
  content: "\f1dd";
}
.fa-sliders:before {
  content: "\f1de";
}
.fa-share-alt:before {
  content: "\f1e0";
}
.fa-share-alt-square:before {
  content: "\f1e1";
}
.fa-bomb:before {
  content: "\f1e2";
}
.fa-soccer-ball-o:before,
.fa-futbol-o:before {
  content: "\f1e3";
}
.fa-tty:before {
  content: "\f1e4";
}
.fa-binoculars:before {
  content: "\f1e5";
}
.fa-plug:before {
  content: "\f1e6";
}
.fa-slideshare:before {
  content: "\f1e7";
}
.fa-twitch:before {
  content: "\f1e8";
}
.fa-yelp:before {
  content: "\f1e9";
}
.fa-newspaper-o:before {
  content: "\f1ea";
}
.fa-wifi:before {
  content: "\f1eb";
}
.fa-calculator:before {
  content: "\f1ec";
}
.fa-paypal:before {
  content: "\f1ed";
}
.fa-google-wallet:before {
  content: "\f1ee";
}
.fa-cc-visa:before {
  content: "\f1f0";
}
.fa-cc-mastercard:before {
  content: "\f1f1";
}
.fa-cc-discover:before {
  content: "\f1f2";
}
.fa-cc-amex:before {
  content: "\f1f3";
}
.fa-cc-paypal:before {
  content: "\f1f4";
}
.fa-cc-stripe:before {
  content: "\f1f5";
}
.fa-bell-slash:before {
  content: "\f1f6";
}
.fa-bell-slash-o:before {
  content: "\f1f7";
}
.fa-trash:before {
  content: "\f1f8";
}
.fa-copyright:before {
  content: "\f1f9";
}
.fa-at:before {
  content: "\f1fa";
}
.fa-eyedropper:before {
  content: "\f1fb";
}
.fa-paint-brush:before {
  content: "\f1fc";
}
.fa-birthday-cake:before {
  content: "\f1fd";
}
.fa-area-chart:before {
  content: "\f1fe";
}
.fa-pie-chart:before {
  content: "\f200";
}
.fa-line-chart:before {
  content: "\f201";
}
.fa-lastfm:before {
  content: "\f202";
}
.fa-lastfm-square:before {
  content: "\f203";
}
.fa-toggle-off:before {
  content: "\f204";
}
.fa-toggle-on:before {
  content: "\f205";
}
.fa-bicycle:before {
  content: "\f206";
}
.fa-bus:before {
  content: "\f207";
}
.fa-ioxhost:before {
  content: "\f208";
}
.fa-angellist:before {
  content: "\f209";
}
.fa-cc:before {
  content: "\f20a";
}
.fa-shekel:before,
.fa-sheqel:before,
.fa-ils:before {
  content: "\f20b";
}
.fa-meanpath:before {
  content: "\f20c";
}
.fa-buysellads:before {
  content: "\f20d";
}
.fa-connectdevelop:before {
  content: "\f20e";
}
.fa-dashcube:before {
  content: "\f210";
}
.fa-forumbee:before {
  content: "\f211";
}
.fa-leanpub:before {
  content: "\f212";
}
.fa-sellsy:before {
  content: "\f213";
}
.fa-shirtsinbulk:before {
  content: "\f214";
}
.fa-simplybuilt:before {
  content: "\f215";
}
.fa-skyatlas:before {
  content: "\f216";
}
.fa-cart-plus:before {
  content: "\f217";
}
.fa-cart-arrow-down:before {
  content: "\f218";
}
.fa-diamond:before {
  content: "\f219";
}
.fa-ship:before {
  content: "\f21a";
}
.fa-user-secret:before {
  content: "\f21b";
}
.fa-motorcycle:before {
  content: "\f21c";
}
.fa-street-view:before {
  content: "\f21d";
}
.fa-heartbeat:before {
  content: "\f21e";
}
.fa-venus:before {
  content: "\f221";
}
.fa-mars:before {
  content: "\f222";
}
.fa-mercury:before {
  content: "\f223";
}
.fa-intersex:before,
.fa-transgender:before {
  content: "\f224";
}
.fa-transgender-alt:before {
  content: "\f225";
}
.fa-venus-double:before {
  content: "\f226";
}
.fa-mars-double:before {
  content: "\f227";
}
.fa-venus-mars:before {
  content: "\f228";
}
.fa-mars-stroke:before {
  content: "\f229";
}
.fa-mars-stroke-v:before {
  content: "\f22a";
}
.fa-mars-stroke-h:before {
  content: "\f22b";
}
.fa-neuter:before {
  content: "\f22c";
}
.fa-genderless:before {
  content: "\f22d";
}
.fa-facebook-official:before {
  content: "\f230";
}
.fa-pinterest-p:before {
  content: "\f231";
}
.fa-whatsapp:before {
  content: "\f232";
}
.fa-server:before {
  content: "\f233";
}
.fa-user-plus:before {
  content: "\f234";
}
.fa-user-times:before {
  content: "\f235";
}
.fa-hotel:before,
.fa-bed:before {
  content: "\f236";
}
.fa-viacoin:before {
  content: "\f237";
}
.fa-train:before {
  content: "\f238";
}
.fa-subway:before {
  content: "\f239";
}
.fa-medium:before {
  content: "\f23a";
}
.fa-yc:before,
.fa-y-combinator:before {
  content: "\f23b";
}
.fa-optin-monster:before {
  content: "\f23c";
}
.fa-opencart:before {
  content: "\f23d";
}
.fa-expeditedssl:before {
  content: "\f23e";
}
.fa-battery-4:before,
.fa-battery:before,
.fa-battery-full:before {
  content: "\f240";
}
.fa-battery-3:before,
.fa-battery-three-quarters:before {
  content: "\f241";
}
.fa-battery-2:before,
.fa-battery-half:before {
  content: "\f242";
}
.fa-battery-1:before,
.fa-battery-quarter:before {
  content: "\f243";
}
.fa-battery-0:before,
.fa-battery-empty:before {
  content: "\f244";
}
.fa-mouse-pointer:before {
  content: "\f245";
}
.fa-i-cursor:before {
  content: "\f246";
}
.fa-object-group:before {
  content: "\f247";
}
.fa-object-ungroup:before {
  content: "\f248";
}
.fa-sticky-note:before {
  content: "\f249";
}
.fa-sticky-note-o:before {
  content: "\f24a";
}
.fa-cc-jcb:before {
  content: "\f24b";
}
.fa-cc-diners-club:before {
  content: "\f24c";
}
.fa-clone:before {
  content: "\f24d";
}
.fa-balance-scale:before {
  content: "\f24e";
}
.fa-hourglass-o:before {
  content: "\f250";
}
.fa-hourglass-1:before,
.fa-hourglass-start:before {
  content: "\f251";
}
.fa-hourglass-2:before,
.fa-hourglass-half:before {
  content: "\f252";
}
.fa-hourglass-3:before,
.fa-hourglass-end:before {
  content: "\f253";
}
.fa-hourglass:before {
  content: "\f254";
}
.fa-hand-grab-o:before,
.fa-hand-rock-o:before {
  content: "\f255";
}
.fa-hand-stop-o:before,
.fa-hand-paper-o:before {
  content: "\f256";
}
.fa-hand-scissors-o:before {
  content: "\f257";
}
.fa-hand-lizard-o:before {
  content: "\f258";
}
.fa-hand-spock-o:before {
  content: "\f259";
}
.fa-hand-pointer-o:before {
  content: "\f25a";
}
.fa-hand-peace-o:before {
  content: "\f25b";
}
.fa-trademark:before {
  content: "\f25c";
}
.fa-registered:before {
  content: "\f25d";
}
.fa-creative-commons:before {
  content: "\f25e";
}
.fa-gg:before {
  content: "\f260";
}
.fa-gg-circle:before {
  content: "\f261";
}
.fa-tripadvisor:before {
  content: "\f262";
}
.fa-odnoklassniki:before {
  content: "\f263";
}
.fa-odnoklassniki-square:before {
  content: "\f264";
}
.fa-get-pocket:before {
  content: "\f265";
}
.fa-wikipedia-w:before {
  content: "\f266";
}
.fa-safari:before {
  content: "\f267";
}
.fa-chrome:before {
  content: "\f268";
}
.fa-firefox:before {
  content: "\f269";
}
.fa-opera:before {
  content: "\f26a";
}
.fa-internet-explorer:before {
  content: "\f26b";
}
.fa-tv:before,
.fa-television:before {
  content: "\f26c";
}
.fa-contao:before {
  content: "\f26d";
}
.fa-500px:before {
  content: "\f26e";
}
.fa-amazon:before {
  content: "\f270";
}
.fa-calendar-plus-o:before {
  content: "\f271";
}
.fa-calendar-minus-o:before {
  content: "\f272";
}
.fa-calendar-times-o:before {
  content: "\f273";
}
.fa-calendar-check-o:before {
  content: "\f274";
}
.fa-industry:before {
  content: "\f275";
}
.fa-map-pin:before {
  content: "\f276";
}
.fa-map-signs:before {
  content: "\f277";
}
.fa-map-o:before {
  content: "\f278";
}
.fa-map:before {
  content: "\f279";
}
.fa-commenting:before {
  content: "\f27a";
}
.fa-commenting-o:before {
  content: "\f27b";
}
.fa-houzz:before {
  content: "\f27c";
}
.fa-vimeo:before {
  content: "\f27d";
}
.fa-black-tie:before {
  content: "\f27e";
}
.fa-fonticons:before {
  content: "\f280";
}
.fa-reddit-alien:before {
  content: "\f281";
}
.fa-edge:before {
  content: "\f282";
}
.fa-credit-card-alt:before {
  content: "\f283";
}
.fa-codiepie:before {
  content: "\f284";
}
.fa-modx:before {
  content: "\f285";
}
.fa-fort-awesome:before {
  content: "\f286";
}
.fa-usb:before {
  content: "\f287";
}
.fa-product-hunt:before {
  content: "\f288";
}
.fa-mixcloud:before {
  content: "\f289";
}
.fa-scribd:before {
  content: "\f28a";
}
.fa-pause-circle:before {
  content: "\f28b";
}
.fa-pause-circle-o:before {
  content: "\f28c";
}
.fa-stop-circle:before {
  content: "\f28d";
}
.fa-stop-circle-o:before {
  content: "\f28e";
}
.fa-shopping-bag:before {
  content: "\f290";
}
.fa-shopping-basket:before {
  content: "\f291";
}
.fa-hashtag:before {
  content: "\f292";
}
.fa-bluetooth:before {
  content: "\f293";
}
.fa-bluetooth-b:before {
  content: "\f294";
}
.fa-percent:before {
  content: "\f295";
}
.fa-gitlab:before {
  content: "\f296";
}
.fa-wpbeginner:before {
  content: "\f297";
}
.fa-wpforms:before {
  content: "\f298";
}
.fa-envira:before {
  content: "\f299";
}
.fa-universal-access:before {
  content: "\f29a";
}
.fa-wheelchair-alt:before {
  content: "\f29b";
}
.fa-question-circle-o:before {
  content: "\f29c";
}
.fa-blind:before {
  content: "\f29d";
}
.fa-audio-description:before {
  content: "\f29e";
}
.fa-volume-control-phone:before {
  content: "\f2a0";
}
.fa-braille:before {
  content: "\f2a1";
}
.fa-assistive-listening-systems:before {
  content: "\f2a2";
}
.fa-asl-interpreting:before,
.fa-american-sign-language-interpreting:before {
  content: "\f2a3";
}
.fa-deafness:before,
.fa-hard-of-hearing:before,
.fa-deaf:before {
  content: "\f2a4";
}
.fa-glide:before {
  content: "\f2a5";
}
.fa-glide-g:before {
  content: "\f2a6";
}
.fa-signing:before,
.fa-sign-language:before {
  content: "\f2a7";
}
.fa-low-vision:before {
  content: "\f2a8";
}
.fa-viadeo:before {
  content: "\f2a9";
}
.fa-viadeo-square:before {
  content: "\f2aa";
}
.fa-snapchat:before {
  content: "\f2ab";
}
.fa-snapchat-ghost:before {
  content: "\f2ac";
}
.fa-snapchat-square:before {
  content: "\f2ad";
}
.fa-pied-piper:before {
  content: "\f2ae";
}
.fa-first-order:before {
  content: "\f2b0";
}
.fa-yoast:before {
  content: "\f2b1";
}
.fa-themeisle:before {
  content: "\f2b2";
}
.fa-google-plus-circle:before,
.fa-google-plus-official:before {
  content: "\f2b3";
}
.fa-fa:before,
.fa-font-awesome:before {
  content: "\f2b4";
}
.fa-handshake-o:before {
  content: "\f2b5";
}
.fa-envelope-open:before {
  content: "\f2b6";
}
.fa-envelope-open-o:before {
  content: "\f2b7";
}
.fa-linode:before {
  content: "\f2b8";
}
.fa-address-book:before {
  content: "\f2b9";
}
.fa-address-book-o:before {
  content: "\f2ba";
}
.fa-vcard:before,
.fa-address-card:before {
  content: "\f2bb";
}
.fa-vcard-o:before,
.fa-address-card-o:before {
  content: "\f2bc";
}
.fa-user-circle:before {
  content: "\f2bd";
}
.fa-user-circle-o:before {
  content: "\f2be";
}
.fa-user-o:before {
  content: "\f2c0";
}
.fa-id-badge:before {
  content: "\f2c1";
}
.fa-drivers-license:before,
.fa-id-card:before {
  content: "\f2c2";
}
.fa-drivers-license-o:before,
.fa-id-card-o:before {
  content: "\f2c3";
}
.fa-quora:before {
  content: "\f2c4";
}
.fa-free-code-camp:before {
  content: "\f2c5";
}
.fa-telegram:before {
  content: "\f2c6";
}
.fa-thermometer-4:before,
.fa-thermometer:before,
.fa-thermometer-full:before {
  content: "\f2c7";
}
.fa-thermometer-3:before,
.fa-thermometer-three-quarters:before {
  content: "\f2c8";
}
.fa-thermometer-2:before,
.fa-thermometer-half:before {
  content: "\f2c9";
}
.fa-thermometer-1:before,
.fa-thermometer-quarter:before {
  content: "\f2ca";
}
.fa-thermometer-0:before,
.fa-thermometer-empty:before {
  content: "\f2cb";
}
.fa-shower:before {
  content: "\f2cc";
}
.fa-bathtub:before,
.fa-s15:before,
.fa-bath:before {
  content: "\f2cd";
}
.fa-podcast:before {
  content: "\f2ce";
}
.fa-window-maximize:before {
  content: "\f2d0";
}
.fa-window-minimize:before {
  content: "\f2d1";
}
.fa-window-restore:before {
  content: "\f2d2";
}
.fa-times-rectangle:before,
.fa-window-close:before {
  content: "\f2d3";
}
.fa-times-rectangle-o:before,
.fa-window-close-o:before {
  content: "\f2d4";
}
.fa-bandcamp:before {
  content: "\f2d5";
}
.fa-grav:before {
  content: "\f2d6";
}
.fa-etsy:before {
  content: "\f2d7";
}
.fa-imdb:before {
  content: "\f2d8";
}
.fa-ravelry:before {
  content: "\f2d9";
}
.fa-eercast:before {
  content: "\f2da";
}
.fa-microchip:before {
  content: "\f2db";
}
.fa-snowflake-o:before {
  content: "\f2dc";
}
.fa-superpowers:before {
  content: "\f2dd";
}
.fa-wpexplorer:before {
  content: "\f2de";
}
.fa-meetup:before {
  content: "\f2e0";
}
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  border: 0;
}
.sr-only-focusable:active,
.sr-only-focusable:focus {
  position: static;
  width: auto;
  height: auto;
  margin: 0;
  overflow: visible;
  clip: auto;
}
.sr-only-focusable:active,
.sr-only-focusable:focus {
  position: static;
  width: auto;
  height: auto;
  margin: 0;
  overflow: visible;
  clip: auto;
}
/*!
*
* IPython base
*
*/
.modal.fade .modal-dialog {
  -webkit-transform: translate(0, 0);
  -ms-transform: translate(0, 0);
  -o-transform: translate(0, 0);
  transform: translate(0, 0);
}
code {
  color: #000;
}
pre {
  font-size: inherit;
  line-height: inherit;
}
label {
  font-weight: normal;
}
/* Make the page background atleast 100% the height of the view port */
/* Make the page itself atleast 70% the height of the view port */
.border-box-sizing {
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
.corner-all {
  border-radius: 2px;
}
.no-padding {
  padding: 0px;
}
/* Flexible box model classes */
/* Taken from Alex Russell http://infrequently.org/2009/08/css-3-progress/ */
/* This file is a compatability layer.  It allows the usage of flexible box 
model layouts accross multiple browsers, including older browsers.  The newest,
universal implementation of the flexible box model is used when available (see
`Modern browsers` comments below).  Browsers that are known to implement this 
new spec completely include:

    Firefox 28.0+
    Chrome 29.0+
    Internet Explorer 11+ 
    Opera 17.0+

Browsers not listed, including Safari, are supported via the styling under the
`Old browsers` comments below.
*/
.hbox {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
.hbox > * {
  /* Old browsers */
  -webkit-box-flex: 0;
  -moz-box-flex: 0;
  box-flex: 0;
  /* Modern browsers */
  flex: none;
}
.vbox {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
}
.vbox > * {
  /* Old browsers */
  -webkit-box-flex: 0;
  -moz-box-flex: 0;
  box-flex: 0;
  /* Modern browsers */
  flex: none;
}
.hbox.reverse,
.vbox.reverse,
.reverse {
  /* Old browsers */
  -webkit-box-direction: reverse;
  -moz-box-direction: reverse;
  box-direction: reverse;
  /* Modern browsers */
  flex-direction: row-reverse;
}
.hbox.box-flex0,
.vbox.box-flex0,
.box-flex0 {
  /* Old browsers */
  -webkit-box-flex: 0;
  -moz-box-flex: 0;
  box-flex: 0;
  /* Modern browsers */
  flex: none;
  width: auto;
}
.hbox.box-flex1,
.vbox.box-flex1,
.box-flex1 {
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
.hbox.box-flex,
.vbox.box-flex,
.box-flex {
  /* Old browsers */
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
.hbox.box-flex2,
.vbox.box-flex2,
.box-flex2 {
  /* Old browsers */
  -webkit-box-flex: 2;
  -moz-box-flex: 2;
  box-flex: 2;
  /* Modern browsers */
  flex: 2;
}
.box-group1 {
  /*  Deprecated */
  -webkit-box-flex-group: 1;
  -moz-box-flex-group: 1;
  box-flex-group: 1;
}
.box-group2 {
  /* Deprecated */
  -webkit-box-flex-group: 2;
  -moz-box-flex-group: 2;
  box-flex-group: 2;
}
.hbox.start,
.vbox.start,
.start {
  /* Old browsers */
  -webkit-box-pack: start;
  -moz-box-pack: start;
  box-pack: start;
  /* Modern browsers */
  justify-content: flex-start;
}
.hbox.end,
.vbox.end,
.end {
  /* Old browsers */
  -webkit-box-pack: end;
  -moz-box-pack: end;
  box-pack: end;
  /* Modern browsers */
  justify-content: flex-end;
}
.hbox.center,
.vbox.center,
.center {
  /* Old browsers */
  -webkit-box-pack: center;
  -moz-box-pack: center;
  box-pack: center;
  /* Modern browsers */
  justify-content: center;
}
.hbox.baseline,
.vbox.baseline,
.baseline {
  /* Old browsers */
  -webkit-box-pack: baseline;
  -moz-box-pack: baseline;
  box-pack: baseline;
  /* Modern browsers */
  justify-content: baseline;
}
.hbox.stretch,
.vbox.stretch,
.stretch {
  /* Old browsers */
  -webkit-box-pack: stretch;
  -moz-box-pack: stretch;
  box-pack: stretch;
  /* Modern browsers */
  justify-content: stretch;
}
.hbox.align-start,
.vbox.align-start,
.align-start {
  /* Old browsers */
  -webkit-box-align: start;
  -moz-box-align: start;
  box-align: start;
  /* Modern browsers */
  align-items: flex-start;
}
.hbox.align-end,
.vbox.align-end,
.align-end {
  /* Old browsers */
  -webkit-box-align: end;
  -moz-box-align: end;
  box-align: end;
  /* Modern browsers */
  align-items: flex-end;
}
.hbox.align-center,
.vbox.align-center,
.align-center {
  /* Old browsers */
  -webkit-box-align: center;
  -moz-box-align: center;
  box-align: center;
  /* Modern browsers */
  align-items: center;
}
.hbox.align-baseline,
.vbox.align-baseline,
.align-baseline {
  /* Old browsers */
  -webkit-box-align: baseline;
  -moz-box-align: baseline;
  box-align: baseline;
  /* Modern browsers */
  align-items: baseline;
}
.hbox.align-stretch,
.vbox.align-stretch,
.align-stretch {
  /* Old browsers */
  -webkit-box-align: stretch;
  -moz-box-align: stretch;
  box-align: stretch;
  /* Modern browsers */
  align-items: stretch;
}
div.error {
  margin: 2em;
  text-align: center;
}
div.error > h1 {
  font-size: 500%;
  line-height: normal;
}
div.error > p {
  font-size: 200%;
  line-height: normal;
}
div.traceback-wrapper {
  text-align: left;
  max-width: 800px;
  margin: auto;
}
div.traceback-wrapper pre.traceback {
  max-height: 600px;
  overflow: auto;
}
/**
 * Primary styles
 *
 * Author: Jupyter Development Team
 */
body {
  background-color: #fff;
  /* This makes sure that the body covers the entire window and needs to
       be in a different element than the display: box in wrapper below */
  position: absolute;
  left: 0px;
  right: 0px;
  top: 0px;
  bottom: 0px;
  overflow: visible;
}
body > #header {
  /* Initially hidden to prevent FLOUC */
  display: none;
  background-color: #fff;
  /* Display over codemirror */
  position: relative;
  z-index: 100;
}
body > #header #header-container {
  display: flex;
  flex-direction: row;
  justify-content: space-between;
  padding: 5px;
  padding-bottom: 5px;
  padding-top: 5px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
body > #header .header-bar {
  width: 100%;
  height: 1px;
  background: #e7e7e7;
  margin-bottom: -1px;
}
@media print {
  body > #header {
    display: none !important;
  }
}
#header-spacer {
  width: 100%;
  visibility: hidden;
}
@media print {
  #header-spacer {
    display: none;
  }
}
#ipython_notebook {
  padding-left: 0px;
  padding-top: 1px;
  padding-bottom: 1px;
}
[dir="rtl"] #ipython_notebook {
  margin-right: 10px;
  margin-left: 0;
}
[dir="rtl"] #ipython_notebook.pull-left {
  float: right !important;
  float: right;
}
.flex-spacer {
  flex: 1;
}
#noscript {
  width: auto;
  padding-top: 16px;
  padding-bottom: 16px;
  text-align: center;
  font-size: 22px;
  color: red;
  font-weight: bold;
}
#ipython_notebook img {
  height: 28px;
}
#site {
  width: 100%;
  display: none;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  overflow: auto;
}
@media print {
  #site {
    height: auto !important;
  }
}
/* Smaller buttons */
.ui-button .ui-button-text {
  padding: 0.2em 0.8em;
  font-size: 77%;
}
input.ui-button {
  padding: 0.3em 0.9em;
}
span#kernel_logo_widget {
  margin: 0 10px;
}
span#login_widget {
  float: right;
}
[dir="rtl"] span#login_widget {
  float: left;
}
span#login_widget > .button,
#logout {
  color: #333;
  background-color: #fff;
  border-color: #ccc;
}
span#login_widget > .button:focus,
#logout:focus,
span#login_widget > .button.focus,
#logout.focus {
  color: #333;
  background-color: #e6e6e6;
  border-color: #8c8c8c;
}
span#login_widget > .button:hover,
#logout:hover {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
span#login_widget > .button:active,
#logout:active,
span#login_widget > .button.active,
#logout.active,
.open > .dropdown-togglespan#login_widget > .button,
.open > .dropdown-toggle#logout {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
span#login_widget > .button:active:hover,
#logout:active:hover,
span#login_widget > .button.active:hover,
#logout.active:hover,
.open > .dropdown-togglespan#login_widget > .button:hover,
.open > .dropdown-toggle#logout:hover,
span#login_widget > .button:active:focus,
#logout:active:focus,
span#login_widget > .button.active:focus,
#logout.active:focus,
.open > .dropdown-togglespan#login_widget > .button:focus,
.open > .dropdown-toggle#logout:focus,
span#login_widget > .button:active.focus,
#logout:active.focus,
span#login_widget > .button.active.focus,
#logout.active.focus,
.open > .dropdown-togglespan#login_widget > .button.focus,
.open > .dropdown-toggle#logout.focus {
  color: #333;
  background-color: #d4d4d4;
  border-color: #8c8c8c;
}
span#login_widget > .button:active,
#logout:active,
span#login_widget > .button.active,
#logout.active,
.open > .dropdown-togglespan#login_widget > .button,
.open > .dropdown-toggle#logout {
  background-image: none;
}
span#login_widget > .button.disabled:hover,
#logout.disabled:hover,
span#login_widget > .button[disabled]:hover,
#logout[disabled]:hover,
fieldset[disabled] span#login_widget > .button:hover,
fieldset[disabled] #logout:hover,
span#login_widget > .button.disabled:focus,
#logout.disabled:focus,
span#login_widget > .button[disabled]:focus,
#logout[disabled]:focus,
fieldset[disabled] span#login_widget > .button:focus,
fieldset[disabled] #logout:focus,
span#login_widget > .button.disabled.focus,
#logout.disabled.focus,
span#login_widget > .button[disabled].focus,
#logout[disabled].focus,
fieldset[disabled] span#login_widget > .button.focus,
fieldset[disabled] #logout.focus {
  background-color: #fff;
  border-color: #ccc;
}
span#login_widget > .button .badge,
#logout .badge {
  color: #fff;
  background-color: #333;
}
.nav-header {
  text-transform: none;
}
#header > span {
  margin-top: 10px;
}
.modal_stretch .modal-dialog {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  min-height: 80vh;
}
.modal_stretch .modal-dialog .modal-body {
  max-height: calc(100vh - 200px);
  overflow: auto;
  flex: 1;
}
.modal-header {
  cursor: move;
}
@media (min-width: 768px) {
  .modal .modal-dialog {
    width: 700px;
  }
}
@media (min-width: 768px) {
  select.form-control {
    margin-left: 12px;
    margin-right: 12px;
  }
}
/*!
*
* IPython auth
*
*/
.center-nav {
  display: inline-block;
  margin-bottom: -4px;
}
[dir="rtl"] .center-nav form.pull-left {
  float: right !important;
  float: right;
}
[dir="rtl"] .center-nav .navbar-text {
  float: right;
}
[dir="rtl"] .navbar-inner {
  text-align: right;
}
[dir="rtl"] div.text-left {
  text-align: right;
}
/*!
*
* IPython tree view
*
*/
/* We need an invisible input field on top of the sentense*/
/* "Drag file onto the list ..." */
.alternate_upload {
  background-color: none;
  display: inline;
}
.alternate_upload.form {
  padding: 0;
  margin: 0;
}
.alternate_upload input.fileinput {
  position: absolute;
  display: block;
  width: 100%;
  height: 100%;
  overflow: hidden;
  cursor: pointer;
  opacity: 0;
  z-index: 2;
}
.alternate_upload .btn-xs > input.fileinput {
  margin: -1px -5px;
}
.alternate_upload .btn-upload {
  position: relative;
  height: 22px;
}
::-webkit-file-upload-button {
  cursor: pointer;
}
/**
 * Primary styles
 *
 * Author: Jupyter Development Team
 */
ul#tabs {
  margin-bottom: 4px;
}
ul#tabs a {
  padding-top: 6px;
  padding-bottom: 4px;
}
[dir="rtl"] ul#tabs.nav-tabs > li {
  float: right;
}
[dir="rtl"] ul#tabs.nav.nav-tabs {
  padding-right: 0;
}
ul.breadcrumb a:focus,
ul.breadcrumb a:hover {
  text-decoration: none;
}
ul.breadcrumb i.icon-home {
  font-size: 16px;
  margin-right: 4px;
}
ul.breadcrumb span {
  color: #5e5e5e;
}
.list_toolbar {
  padding: 4px 0 4px 0;
  vertical-align: middle;
}
.list_toolbar .tree-buttons {
  padding-top: 1px;
}
[dir="rtl"] .list_toolbar .tree-buttons .pull-right {
  float: left !important;
  float: left;
}
[dir="rtl"] .list_toolbar .col-sm-4,
[dir="rtl"] .list_toolbar .col-sm-8 {
  float: right;
}
.dynamic-buttons {
  padding-top: 3px;
  display: inline-block;
}
.list_toolbar [class*="span"] {
  min-height: 24px;
}
.list_header {
  font-weight: bold;
  background-color: #EEE;
}
.list_placeholder {
  font-weight: bold;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 7px;
  padding-right: 7px;
}
.list_container {
  margin-top: 4px;
  margin-bottom: 20px;
  border: 1px solid #ddd;
  border-radius: 2px;
}
.list_container > div {
  border-bottom: 1px solid #ddd;
}
.list_container > div:hover .list-item {
  background-color: red;
}
.list_container > div:last-child {
  border: none;
}
.list_item:hover .list_item {
  background-color: #ddd;
}
.list_item a {
  text-decoration: none;
}
.list_item:hover {
  background-color: #fafafa;
}
.list_header > div,
.list_item > div {
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 7px;
  padding-right: 7px;
  line-height: 22px;
}
.list_header > div input,
.list_item > div input {
  margin-right: 7px;
  margin-left: 14px;
  vertical-align: text-bottom;
  line-height: 22px;
  position: relative;
  top: -1px;
}
.list_header > div .item_link,
.list_item > div .item_link {
  margin-left: -1px;
  vertical-align: baseline;
  line-height: 22px;
}
[dir="rtl"] .list_item > div input {
  margin-right: 0;
}
.new-file input[type=checkbox] {
  visibility: hidden;
}
.item_name {
  line-height: 22px;
  height: 24px;
}
.item_icon {
  font-size: 14px;
  color: #5e5e5e;
  margin-right: 7px;
  margin-left: 7px;
  line-height: 22px;
  vertical-align: baseline;
}
.item_modified {
  margin-right: 7px;
  margin-left: 7px;
}
[dir="rtl"] .item_modified.pull-right {
  float: left !important;
  float: left;
}
.item_buttons {
  line-height: 1em;
  margin-left: -5px;
}
.item_buttons .btn,
.item_buttons .btn-group,
.item_buttons .input-group {
  float: left;
}
.item_buttons > .btn,
.item_buttons > .btn-group,
.item_buttons > .input-group {
  margin-left: 5px;
}
.item_buttons .btn {
  min-width: 13ex;
}
.item_buttons .running-indicator {
  padding-top: 4px;
  color: #5cb85c;
}
.item_buttons .kernel-name {
  padding-top: 4px;
  color: #5bc0de;
  margin-right: 7px;
  float: left;
}
[dir="rtl"] .item_buttons.pull-right {
  float: left !important;
  float: left;
}
[dir="rtl"] .item_buttons .kernel-name {
  margin-left: 7px;
  float: right;
}
.toolbar_info {
  height: 24px;
  line-height: 24px;
}
.list_item input:not([type=checkbox]) {
  padding-top: 3px;
  padding-bottom: 3px;
  height: 22px;
  line-height: 14px;
  margin: 0px;
}
.highlight_text {
  color: blue;
}
#project_name {
  display: inline-block;
  padding-left: 7px;
  margin-left: -2px;
}
#project_name > .breadcrumb {
  padding: 0px;
  margin-bottom: 0px;
  background-color: transparent;
  font-weight: bold;
}
.sort_button {
  display: inline-block;
  padding-left: 7px;
}
[dir="rtl"] .sort_button.pull-right {
  float: left !important;
  float: left;
}
#tree-selector {
  padding-right: 0px;
}
#button-select-all {
  min-width: 50px;
}
[dir="rtl"] #button-select-all.btn {
  float: right ;
}
#select-all {
  margin-left: 7px;
  margin-right: 2px;
  margin-top: 2px;
  height: 16px;
}
[dir="rtl"] #select-all.pull-left {
  float: right !important;
  float: right;
}
.menu_icon {
  margin-right: 2px;
}
.tab-content .row {
  margin-left: 0px;
  margin-right: 0px;
}
.folder_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f114";
}
.folder_icon:before.fa-pull-left {
  margin-right: .3em;
}
.folder_icon:before.fa-pull-right {
  margin-left: .3em;
}
.folder_icon:before.pull-left {
  margin-right: .3em;
}
.folder_icon:before.pull-right {
  margin-left: .3em;
}
.notebook_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f02d";
  position: relative;
  top: -1px;
}
.notebook_icon:before.fa-pull-left {
  margin-right: .3em;
}
.notebook_icon:before.fa-pull-right {
  margin-left: .3em;
}
.notebook_icon:before.pull-left {
  margin-right: .3em;
}
.notebook_icon:before.pull-right {
  margin-left: .3em;
}
.running_notebook_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f02d";
  position: relative;
  top: -1px;
  color: #5cb85c;
}
.running_notebook_icon:before.fa-pull-left {
  margin-right: .3em;
}
.running_notebook_icon:before.fa-pull-right {
  margin-left: .3em;
}
.running_notebook_icon:before.pull-left {
  margin-right: .3em;
}
.running_notebook_icon:before.pull-right {
  margin-left: .3em;
}
.file_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f016";
  position: relative;
  top: -2px;
}
.file_icon:before.fa-pull-left {
  margin-right: .3em;
}
.file_icon:before.fa-pull-right {
  margin-left: .3em;
}
.file_icon:before.pull-left {
  margin-right: .3em;
}
.file_icon:before.pull-right {
  margin-left: .3em;
}
#notebook_toolbar .pull-right {
  padding-top: 0px;
  margin-right: -1px;
}
ul#new-menu {
  left: auto;
  right: 0;
}
#new-menu .dropdown-header {
  font-size: 10px;
  border-bottom: 1px solid #e5e5e5;
  padding: 0 0 3px;
  margin: -3px 20px 0;
}
.kernel-menu-icon {
  padding-right: 12px;
  width: 24px;
  content: "\f096";
}
.kernel-menu-icon:before {
  content: "\f096";
}
.kernel-menu-icon-current:before {
  content: "\f00c";
}
#tab_content {
  padding-top: 20px;
}
#running .panel-group .panel {
  margin-top: 3px;
  margin-bottom: 1em;
}
#running .panel-group .panel .panel-heading {
  background-color: #EEE;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 7px;
  padding-right: 7px;
  line-height: 22px;
}
#running .panel-group .panel .panel-heading a:focus,
#running .panel-group .panel .panel-heading a:hover {
  text-decoration: none;
}
#running .panel-group .panel .panel-body {
  padding: 0px;
}
#running .panel-group .panel .panel-body .list_container {
  margin-top: 0px;
  margin-bottom: 0px;
  border: 0px;
  border-radius: 0px;
}
#running .panel-group .panel .panel-body .list_container .list_item {
  border-bottom: 1px solid #ddd;
}
#running .panel-group .panel .panel-body .list_container .list_item:last-child {
  border-bottom: 0px;
}
.delete-button {
  display: none;
}
.duplicate-button {
  display: none;
}
.rename-button {
  display: none;
}
.move-button {
  display: none;
}
.download-button {
  display: none;
}
.shutdown-button {
  display: none;
}
.dynamic-instructions {
  display: inline-block;
  padding-top: 4px;
}
/*!
*
* IPython text editor webapp
*
*/
.selected-keymap i.fa {
  padding: 0px 5px;
}
.selected-keymap i.fa:before {
  content: "\f00c";
}
#mode-menu {
  overflow: auto;
  max-height: 20em;
}
.edit_app #header {
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
}
.edit_app #menubar .navbar {
  /* Use a negative 1 bottom margin, so the border overlaps the border of the
    header */
  margin-bottom: -1px;
}
.dirty-indicator {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  width: 20px;
}
.dirty-indicator.fa-pull-left {
  margin-right: .3em;
}
.dirty-indicator.fa-pull-right {
  margin-left: .3em;
}
.dirty-indicator.pull-left {
  margin-right: .3em;
}
.dirty-indicator.pull-right {
  margin-left: .3em;
}
.dirty-indicator-dirty {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  width: 20px;
}
.dirty-indicator-dirty.fa-pull-left {
  margin-right: .3em;
}
.dirty-indicator-dirty.fa-pull-right {
  margin-left: .3em;
}
.dirty-indicator-dirty.pull-left {
  margin-right: .3em;
}
.dirty-indicator-dirty.pull-right {
  margin-left: .3em;
}
.dirty-indicator-clean {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  width: 20px;
}
.dirty-indicator-clean.fa-pull-left {
  margin-right: .3em;
}
.dirty-indicator-clean.fa-pull-right {
  margin-left: .3em;
}
.dirty-indicator-clean.pull-left {
  margin-right: .3em;
}
.dirty-indicator-clean.pull-right {
  margin-left: .3em;
}
.dirty-indicator-clean:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f00c";
}
.dirty-indicator-clean:before.fa-pull-left {
  margin-right: .3em;
}
.dirty-indicator-clean:before.fa-pull-right {
  margin-left: .3em;
}
.dirty-indicator-clean:before.pull-left {
  margin-right: .3em;
}
.dirty-indicator-clean:before.pull-right {
  margin-left: .3em;
}
#filename {
  font-size: 16pt;
  display: table;
  padding: 0px 5px;
}
#current-mode {
  padding-left: 5px;
  padding-right: 5px;
}
#texteditor-backdrop {
  padding-top: 20px;
  padding-bottom: 20px;
}
@media not print {
  #texteditor-backdrop {
    background-color: #EEE;
  }
}
@media print {
  #texteditor-backdrop #texteditor-container .CodeMirror-gutter,
  #texteditor-backdrop #texteditor-container .CodeMirror-gutters {
    background-color: #fff;
  }
}
@media not print {
  #texteditor-backdrop #texteditor-container .CodeMirror-gutter,
  #texteditor-backdrop #texteditor-container .CodeMirror-gutters {
    background-color: #fff;
  }
}
@media not print {
  #texteditor-backdrop #texteditor-container {
    padding: 0px;
    background-color: #fff;
    -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
    box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  }
}
.CodeMirror-dialog {
  background-color: #fff;
}
/*!
*
* IPython notebook
*
*/
/* CSS font colors for translated ANSI escape sequences */
/* The color values are a mix of
   http://www.xcolors.net/dl/baskerville-ivorylight and
   http://www.xcolors.net/dl/euphrasia */
.ansi-black-fg {
  color: #3E424D;
}
.ansi-black-bg {
  background-color: #3E424D;
}
.ansi-black-intense-fg {
  color: #282C36;
}
.ansi-black-intense-bg {
  background-color: #282C36;
}
.ansi-red-fg {
  color: #E75C58;
}
.ansi-red-bg {
  background-color: #E75C58;
}
.ansi-red-intense-fg {
  color: #B22B31;
}
.ansi-red-intense-bg {
  background-color: #B22B31;
}
.ansi-green-fg {
  color: #00A250;
}
.ansi-green-bg {
  background-color: #00A250;
}
.ansi-green-intense-fg {
  color: #007427;
}
.ansi-green-intense-bg {
  background-color: #007427;
}
.ansi-yellow-fg {
  color: #DDB62B;
}
.ansi-yellow-bg {
  background-color: #DDB62B;
}
.ansi-yellow-intense-fg {
  color: #B27D12;
}
.ansi-yellow-intense-bg {
  background-color: #B27D12;
}
.ansi-blue-fg {
  color: #208FFB;
}
.ansi-blue-bg {
  background-color: #208FFB;
}
.ansi-blue-intense-fg {
  color: #0065CA;
}
.ansi-blue-intense-bg {
  background-color: #0065CA;
}
.ansi-magenta-fg {
  color: #D160C4;
}
.ansi-magenta-bg {
  background-color: #D160C4;
}
.ansi-magenta-intense-fg {
  color: #A03196;
}
.ansi-magenta-intense-bg {
  background-color: #A03196;
}
.ansi-cyan-fg {
  color: #60C6C8;
}
.ansi-cyan-bg {
  background-color: #60C6C8;
}
.ansi-cyan-intense-fg {
  color: #258F8F;
}
.ansi-cyan-intense-bg {
  background-color: #258F8F;
}
.ansi-white-fg {
  color: #C5C1B4;
}
.ansi-white-bg {
  background-color: #C5C1B4;
}
.ansi-white-intense-fg {
  color: #A1A6B2;
}
.ansi-white-intense-bg {
  background-color: #A1A6B2;
}
.ansi-default-inverse-fg {
  color: #FFFFFF;
}
.ansi-default-inverse-bg {
  background-color: #000000;
}
.ansi-bold {
  font-weight: bold;
}
.ansi-underline {
  text-decoration: underline;
}
/* The following styles are deprecated an will be removed in a future version */
.ansibold {
  font-weight: bold;
}
.ansi-inverse {
  outline: 0.5px dotted;
}
/* use dark versions for foreground, to improve visibility */
.ansiblack {
  color: black;
}
.ansired {
  color: darkred;
}
.ansigreen {
  color: darkgreen;
}
.ansiyellow {
  color: #c4a000;
}
.ansiblue {
  color: darkblue;
}
.ansipurple {
  color: darkviolet;
}
.ansicyan {
  color: steelblue;
}
.ansigray {
  color: gray;
}
/* and light for background, for the same reason */
.ansibgblack {
  background-color: black;
}
.ansibgred {
  background-color: red;
}
.ansibggreen {
  background-color: green;
}
.ansibgyellow {
  background-color: yellow;
}
.ansibgblue {
  background-color: blue;
}
.ansibgpurple {
  background-color: magenta;
}
.ansibgcyan {
  background-color: cyan;
}
.ansibggray {
  background-color: gray;
}
div.cell {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  border-radius: 2px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  border-width: 1px;
  border-style: solid;
  border-color: transparent;
  width: 100%;
  padding: 5px;
  /* This acts as a spacer between cells, that is outside the border */
  margin: 0px;
  outline: none;
  position: relative;
  overflow: visible;
}
div.cell:before {
  position: absolute;
  display: block;
  top: -1px;
  left: -1px;
  width: 5px;
  height: calc(100% +  2px);
  content: '';
  background: transparent;
}
div.cell.jupyter-soft-selected {
  border-left-color: #E3F2FD;
  border-left-width: 1px;
  padding-left: 5px;
  border-right-color: #E3F2FD;
  border-right-width: 1px;
  background: #E3F2FD;
}
@media print {
  div.cell.jupyter-soft-selected {
    border-color: transparent;
  }
}
div.cell.selected,
div.cell.selected.jupyter-soft-selected {
  border-color: #ababab;
}
div.cell.selected:before,
div.cell.selected.jupyter-soft-selected:before {
  position: absolute;
  display: block;
  top: -1px;
  left: -1px;
  width: 5px;
  height: calc(100% +  2px);
  content: '';
  background: #42A5F5;
}
@media print {
  div.cell.selected,
  div.cell.selected.jupyter-soft-selected {
    border-color: transparent;
  }
}
.edit_mode div.cell.selected {
  border-color: #66BB6A;
}
.edit_mode div.cell.selected:before {
  position: absolute;
  display: block;
  top: -1px;
  left: -1px;
  width: 5px;
  height: calc(100% +  2px);
  content: '';
  background: #66BB6A;
}
@media print {
  .edit_mode div.cell.selected {
    border-color: transparent;
  }
}
.prompt {
  /* This needs to be wide enough for 3 digit prompt numbers: In[100]: */
  min-width: 14ex;
  /* This padding is tuned to match the padding on the CodeMirror editor. */
  padding: 0.4em;
  margin: 0px;
  font-family: monospace;
  text-align: right;
  /* This has to match that of the the CodeMirror class line-height below */
  line-height: 1.21429em;
  /* Don't highlight prompt number selection */
  -webkit-touch-callout: none;
  -webkit-user-select: none;
  -khtml-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
  /* Use default cursor */
  cursor: default;
}
@media (max-width: 540px) {
  .prompt {
    text-align: left;
  }
}
div.inner_cell {
  min-width: 0;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
/* input_area and input_prompt must match in top border and margin for alignment */
div.input_area {
  border: 1px solid #cfcfcf;
  border-radius: 2px;
  background: #f7f7f7;
  line-height: 1.21429em;
}
/* This is needed so that empty prompt areas can collapse to zero height when there
   is no content in the output_subarea and the prompt. The main purpose of this is
   to make sure that empty JavaScript output_subareas have no height. */
div.prompt:empty {
  padding-top: 0;
  padding-bottom: 0;
}
div.unrecognized_cell {
  padding: 5px 5px 5px 0px;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
div.unrecognized_cell .inner_cell {
  border-radius: 2px;
  padding: 5px;
  font-weight: bold;
  color: red;
  border: 1px solid #cfcfcf;
  background: #eaeaea;
}
div.unrecognized_cell .inner_cell a {
  color: inherit;
  text-decoration: none;
}
div.unrecognized_cell .inner_cell a:hover {
  color: inherit;
  text-decoration: none;
}
@media (max-width: 540px) {
  div.unrecognized_cell > div.prompt {
    display: none;
  }
}
div.code_cell {
  /* avoid page breaking on code cells when printing */
}
@media print {
  div.code_cell {
    page-break-inside: avoid;
  }
}
/* any special styling for code cells that are currently running goes here */
div.input {
  page-break-inside: avoid;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
@media (max-width: 540px) {
  div.input {
    /* Old browsers */
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-box-align: stretch;
    display: -moz-box;
    -moz-box-orient: vertical;
    -moz-box-align: stretch;
    display: box;
    box-orient: vertical;
    box-align: stretch;
    /* Modern browsers */
    display: flex;
    flex-direction: column;
    align-items: stretch;
  }
}
/* input_area and input_prompt must match in top border and margin for alignment */
div.input_prompt {
  color: #303F9F;
  border-top: 1px solid transparent;
}
div.input_area > div.highlight {
  margin: 0.4em;
  border: none;
  padding: 0px;
  background-color: transparent;
}
div.input_area > div.highlight > pre {
  margin: 0px;
  border: none;
  padding: 0px;
  background-color: transparent;
}
/* The following gets added to the <head> if it is detected that the user has a
 * monospace font with inconsistent normal/bold/italic height.  See
 * notebookmain.js.  Such fonts will have keywords vertically offset with
 * respect to the rest of the text.  The user should select a better font.
 * See: https://github.com/ipython/ipython/issues/1503
 *
 * .CodeMirror span {
 *      vertical-align: bottom;
 * }
 */
.CodeMirror {
  line-height: 1.21429em;
  /* Changed from 1em to our global default */
  font-size: 14px;
  height: auto;
  /* Changed to auto to autogrow */
  background: none;
  /* Changed from white to allow our bg to show through */
}
.CodeMirror-scroll {
  /*  The CodeMirror docs are a bit fuzzy on if overflow-y should be hidden or visible.*/
  /*  We have found that if it is visible, vertical scrollbars appear with font size changes.*/
  overflow-y: hidden;
  overflow-x: auto;
}
.CodeMirror-lines {
  /* In CM2, this used to be 0.4em, but in CM3 it went to 4px. We need the em value because */
  /* we have set a different line-height and want this to scale with that. */
  /* Note that this should set vertical padding only, since CodeMirror assumes
       that horizontal padding will be set on CodeMirror pre */
  padding: 0.4em 0;
}
.CodeMirror-linenumber {
  padding: 0 8px 0 4px;
}
.CodeMirror-gutters {
  border-bottom-left-radius: 2px;
  border-top-left-radius: 2px;
}
.CodeMirror pre {
  /* In CM3 this went to 4px from 0 in CM2. This sets horizontal padding only,
    use .CodeMirror-lines for vertical */
  padding: 0 0.4em;
  border: 0;
  border-radius: 0;
}
.CodeMirror-cursor {
  border-left: 1.4px solid black;
}
@media screen and (min-width: 2138px) and (max-width: 4319px) {
  .CodeMirror-cursor {
    border-left: 2px solid black;
  }
}
@media screen and (min-width: 4320px) {
  .CodeMirror-cursor {
    border-left: 4px solid black;
  }
}
/*

Original style from softwaremaniacs.org (c) Ivan Sagalaev <Maniac@SoftwareManiacs.Org>
Adapted from GitHub theme

*/
.highlight-base {
  color: #000;
}
.highlight-variable {
  color: #000;
}
.highlight-variable-2 {
  color: #1a1a1a;
}
.highlight-variable-3 {
  color: #333333;
}
.highlight-string {
  color: #BA2121;
}
.highlight-comment {
  color: #408080;
  font-style: italic;
}
.highlight-number {
  color: #080;
}
.highlight-atom {
  color: #88F;
}
.highlight-keyword {
  color: #008000;
  font-weight: bold;
}
.highlight-builtin {
  color: #008000;
}
.highlight-error {
  color: #f00;
}
.highlight-operator {
  color: #AA22FF;
  font-weight: bold;
}
.highlight-meta {
  color: #AA22FF;
}
/* previously not defined, copying from default codemirror */
.highlight-def {
  color: #00f;
}
.highlight-string-2 {
  color: #f50;
}
.highlight-qualifier {
  color: #555;
}
.highlight-bracket {
  color: #997;
}
.highlight-tag {
  color: #170;
}
.highlight-attribute {
  color: #00c;
}
.highlight-header {
  color: blue;
}
.highlight-quote {
  color: #090;
}
.highlight-link {
  color: #00c;
}
/* apply the same style to codemirror */
.cm-s-ipython span.cm-keyword {
  color: #008000;
  font-weight: bold;
}
.cm-s-ipython span.cm-atom {
  color: #88F;
}
.cm-s-ipython span.cm-number {
  color: #080;
}
.cm-s-ipython span.cm-def {
  color: #00f;
}
.cm-s-ipython span.cm-variable {
  color: #000;
}
.cm-s-ipython span.cm-operator {
  color: #AA22FF;
  font-weight: bold;
}
.cm-s-ipython span.cm-variable-2 {
  color: #1a1a1a;
}
.cm-s-ipython span.cm-variable-3 {
  color: #333333;
}
.cm-s-ipython span.cm-comment {
  color: #408080;
  font-style: italic;
}
.cm-s-ipython span.cm-string {
  color: #BA2121;
}
.cm-s-ipython span.cm-string-2 {
  color: #f50;
}
.cm-s-ipython span.cm-meta {
  color: #AA22FF;
}
.cm-s-ipython span.cm-qualifier {
  color: #555;
}
.cm-s-ipython span.cm-builtin {
  color: #008000;
}
.cm-s-ipython span.cm-bracket {
  color: #997;
}
.cm-s-ipython span.cm-tag {
  color: #170;
}
.cm-s-ipython span.cm-attribute {
  color: #00c;
}
.cm-s-ipython span.cm-header {
  color: blue;
}
.cm-s-ipython span.cm-quote {
  color: #090;
}
.cm-s-ipython span.cm-link {
  color: #00c;
}
.cm-s-ipython span.cm-error {
  color: #f00;
}
.cm-s-ipython span.cm-tab {
  background: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAMCAYAAAAkuj5RAAAAAXNSR0IArs4c6QAAAGFJREFUSMft1LsRQFAQheHPowAKoACx3IgEKtaEHujDjORSgWTH/ZOdnZOcM/sgk/kFFWY0qV8foQwS4MKBCS3qR6ixBJvElOobYAtivseIE120FaowJPN75GMu8j/LfMwNjh4HUpwg4LUAAAAASUVORK5CYII=);
  background-position: right;
  background-repeat: no-repeat;
}
div.output_wrapper {
  /* this position must be relative to enable descendents to be absolute within it */
  position: relative;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  z-index: 1;
}
/* class for the output area when it should be height-limited */
div.output_scroll {
  /* ideally, this would be max-height, but FF barfs all over that */
  height: 24em;
  /* FF needs this *and the wrapper* to specify full width, or it will shrinkwrap */
  width: 100%;
  overflow: auto;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.8);
  box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.8);
  display: block;
}
/* output div while it is collapsed */
div.output_collapsed {
  margin: 0px;
  padding: 0px;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
}
div.out_prompt_overlay {
  height: 100%;
  padding: 0px 0.4em;
  position: absolute;
  border-radius: 2px;
}
div.out_prompt_overlay:hover {
  /* use inner shadow to get border that is computed the same on WebKit/FF */
  -webkit-box-shadow: inset 0 0 1px #000;
  box-shadow: inset 0 0 1px #000;
  background: rgba(240, 240, 240, 0.5);
}
div.output_prompt {
  color: #D84315;
}
/* This class is the outer container of all output sections. */
div.output_area {
  padding: 0px;
  page-break-inside: avoid;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
div.output_area .MathJax_Display {
  text-align: left !important;
}
div.output_area .rendered_html table {
  margin-left: 0;
  margin-right: 0;
}
div.output_area .rendered_html img {
  margin-left: 0;
  margin-right: 0;
}
div.output_area img,
div.output_area svg {
  max-width: 100%;
  height: auto;
}
div.output_area img.unconfined,
div.output_area svg.unconfined {
  max-width: none;
}
div.output_area .mglyph > img {
  max-width: none;
}
/* This is needed to protect the pre formating from global settings such
   as that of bootstrap */
.output {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
}
@media (max-width: 540px) {
  div.output_area {
    /* Old browsers */
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-box-align: stretch;
    display: -moz-box;
    -moz-box-orient: vertical;
    -moz-box-align: stretch;
    display: box;
    box-orient: vertical;
    box-align: stretch;
    /* Modern browsers */
    display: flex;
    flex-direction: column;
    align-items: stretch;
  }
}
div.output_area pre {
  margin: 0;
  padding: 1px 0 1px 0;
  border: 0;
  vertical-align: baseline;
  color: black;
  background-color: transparent;
  border-radius: 0;
}
/* This class is for the output subarea inside the output_area and after
   the prompt div. */
div.output_subarea {
  overflow-x: auto;
  padding: 0.4em;
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
  max-width: calc(100% - 14ex);
}
div.output_scroll div.output_subarea {
  overflow-x: visible;
}
/* The rest of the output_* classes are for special styling of the different
   output types */
/* all text output has this class: */
div.output_text {
  text-align: left;
  color: #000;
  /* This has to match that of the the CodeMirror class line-height below */
  line-height: 1.21429em;
}
/* stdout/stderr are 'text' as well as 'stream', but execute_result/error are *not* streams */
div.output_stderr {
  background: #fdd;
  /* very light red background for stderr */
}
div.output_latex {
  text-align: left;
}
/* Empty output_javascript divs should have no height */
div.output_javascript:empty {
  padding: 0;
}
.js-error {
  color: darkred;
}
/* raw_input styles */
div.raw_input_container {
  line-height: 1.21429em;
  padding-top: 5px;
}
pre.raw_input_prompt {
  /* nothing needed here. */
}
input.raw_input {
  font-family: monospace;
  font-size: inherit;
  color: inherit;
  width: auto;
  /* make sure input baseline aligns with prompt */
  vertical-align: baseline;
  /* padding + margin = 0.5em between prompt and cursor */
  padding: 0em 0.25em;
  margin: 0em 0.25em;
}
input.raw_input:focus {
  box-shadow: none;
}
p.p-space {
  margin-bottom: 10px;
}
div.output_unrecognized {
  padding: 5px;
  font-weight: bold;
  color: red;
}
div.output_unrecognized a {
  color: inherit;
  text-decoration: none;
}
div.output_unrecognized a:hover {
  color: inherit;
  text-decoration: none;
}
.rendered_html {
  color: #000;
  /* any extras will just be numbers: */
}
.rendered_html em {
  font-style: italic;
}
.rendered_html strong {
  font-weight: bold;
}
.rendered_html u {
  text-decoration: underline;
}
.rendered_html :link {
  text-decoration: underline;
}
.rendered_html :visited {
  text-decoration: underline;
}
.rendered_html h1 {
  font-size: 185.7%;
  margin: 1.08em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h2 {
  font-size: 157.1%;
  margin: 1.27em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h3 {
  font-size: 128.6%;
  margin: 1.55em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h4 {
  font-size: 100%;
  margin: 2em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h5 {
  font-size: 100%;
  margin: 2em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
  font-style: italic;
}
.rendered_html h6 {
  font-size: 100%;
  margin: 2em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
  font-style: italic;
}
.rendered_html h1:first-child {
  margin-top: 0.538em;
}
.rendered_html h2:first-child {
  margin-top: 0.636em;
}
.rendered_html h3:first-child {
  margin-top: 0.777em;
}
.rendered_html h4:first-child {
  margin-top: 1em;
}
.rendered_html h5:first-child {
  margin-top: 1em;
}
.rendered_html h6:first-child {
  margin-top: 1em;
}
.rendered_html ul:not(.list-inline),
.rendered_html ol:not(.list-inline) {
  padding-left: 2em;
}
.rendered_html ul {
  list-style: disc;
}
.rendered_html ul ul {
  list-style: square;
  margin-top: 0;
}
.rendered_html ul ul ul {
  list-style: circle;
}
.rendered_html ol {
  list-style: decimal;
}
.rendered_html ol ol {
  list-style: upper-alpha;
  margin-top: 0;
}
.rendered_html ol ol ol {
  list-style: lower-alpha;
}
.rendered_html ol ol ol ol {
  list-style: lower-roman;
}
.rendered_html ol ol ol ol ol {
  list-style: decimal;
}
.rendered_html * + ul {
  margin-top: 1em;
}
.rendered_html * + ol {
  margin-top: 1em;
}
.rendered_html hr {
  color: black;
  background-color: black;
}
.rendered_html pre {
  margin: 1em 2em;
  padding: 0px;
  background-color: #fff;
}
.rendered_html code {
  background-color: #eff0f1;
}
.rendered_html p code {
  padding: 1px 5px;
}
.rendered_html pre code {
  background-color: #fff;
}
.rendered_html pre,
.rendered_html code {
  border: 0;
  color: #000;
  font-size: 100%;
}
.rendered_html blockquote {
  margin: 1em 2em;
}
.rendered_html table {
  margin-left: auto;
  margin-right: auto;
  border: none;
  border-collapse: collapse;
  border-spacing: 0;
  color: black;
  font-size: 12px;
  table-layout: fixed;
}
.rendered_html thead {
  border-bottom: 1px solid black;
  vertical-align: bottom;
}
.rendered_html tr,
.rendered_html th,
.rendered_html td {
  text-align: right;
  vertical-align: middle;
  padding: 0.5em 0.5em;
  line-height: normal;
  white-space: normal;
  max-width: none;
  border: none;
}
.rendered_html th {
  font-weight: bold;
}
.rendered_html tbody tr:nth-child(odd) {
  background: #f5f5f5;
}
.rendered_html tbody tr:hover {
  background: rgba(66, 165, 245, 0.2);
}
.rendered_html * + table {
  margin-top: 1em;
}
.rendered_html p {
  text-align: left;
}
.rendered_html * + p {
  margin-top: 1em;
}
.rendered_html img {
  display: block;
  margin-left: auto;
  margin-right: auto;
}
.rendered_html * + img {
  margin-top: 1em;
}
.rendered_html img,
.rendered_html svg {
  max-width: 100%;
  height: auto;
}
.rendered_html img.unconfined,
.rendered_html svg.unconfined {
  max-width: none;
}
.rendered_html .alert {
  margin-bottom: initial;
}
.rendered_html * + .alert {
  margin-top: 1em;
}
[dir="rtl"] .rendered_html p {
  text-align: right;
}
div.text_cell {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
@media (max-width: 540px) {
  div.text_cell > div.prompt {
    display: none;
  }
}
div.text_cell_render {
  /*font-family: "Helvetica Neue", Arial, Helvetica, Geneva, sans-serif;*/
  outline: none;
  resize: none;
  width: inherit;
  border-style: none;
  padding: 0.5em 0.5em 0.5em 0.4em;
  color: #000;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
a.anchor-link:link {
  text-decoration: none;
  padding: 0px 20px;
  visibility: hidden;
}
h1:hover .anchor-link,
h2:hover .anchor-link,
h3:hover .anchor-link,
h4:hover .anchor-link,
h5:hover .anchor-link,
h6:hover .anchor-link {
  visibility: visible;
}
.text_cell.rendered .input_area {
  display: none;
}
.text_cell.rendered .rendered_html {
  overflow-x: auto;
  overflow-y: hidden;
}
.text_cell.rendered .rendered_html tr,
.text_cell.rendered .rendered_html th,
.text_cell.rendered .rendered_html td {
  max-width: none;
}
.text_cell.unrendered .text_cell_render {
  display: none;
}
.text_cell .dropzone .input_area {
  border: 2px dashed #bababa;
  margin: -1px;
}
.cm-header-1,
.cm-header-2,
.cm-header-3,
.cm-header-4,
.cm-header-5,
.cm-header-6 {
  font-weight: bold;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
}
.cm-header-1 {
  font-size: 185.7%;
}
.cm-header-2 {
  font-size: 157.1%;
}
.cm-header-3 {
  font-size: 128.6%;
}
.cm-header-4 {
  font-size: 110%;
}
.cm-header-5 {
  font-size: 100%;
  font-style: italic;
}
.cm-header-6 {
  font-size: 100%;
  font-style: italic;
}
/*!
*
* IPython notebook webapp
*
*/
@media (max-width: 767px) {
  .notebook_app {
    padding-left: 0px;
    padding-right: 0px;
  }
}
#ipython-main-app {
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  height: 100%;
}
div#notebook_panel {
  margin: 0px;
  padding: 0px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  height: 100%;
}
div#notebook {
  font-size: 14px;
  line-height: 20px;
  overflow-y: hidden;
  overflow-x: auto;
  width: 100%;
  /* This spaces the page away from the edge of the notebook area */
  padding-top: 20px;
  margin: 0px;
  outline: none;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  min-height: 100%;
}
@media not print {
  #notebook-container {
    padding: 15px;
    background-color: #fff;
    min-height: 0;
    -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
    box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  }
}
@media print {
  #notebook-container {
    width: 100%;
  }
}
div.ui-widget-content {
  border: 1px solid #ababab;
  outline: none;
}
pre.dialog {
  background-color: #f7f7f7;
  border: 1px solid #ddd;
  border-radius: 2px;
  padding: 0.4em;
  padding-left: 2em;
}
p.dialog {
  padding: 0.2em;
}
/* Word-wrap output correctly.  This is the CSS3 spelling, though Firefox seems
   to not honor it correctly.  Webkit browsers (Chrome, rekonq, Safari) do.
 */
pre,
code,
kbd,
samp {
  white-space: pre-wrap;
}
#fonttest {
  font-family: monospace;
}
p {
  margin-bottom: 0;
}
.end_space {
  min-height: 100px;
  transition: height .2s ease;
}
.notebook_app > #header {
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
}
@media not print {
  .notebook_app {
    background-color: #EEE;
  }
}
kbd {
  border-style: solid;
  border-width: 1px;
  box-shadow: none;
  margin: 2px;
  padding-left: 2px;
  padding-right: 2px;
  padding-top: 1px;
  padding-bottom: 1px;
}
.jupyter-keybindings {
  padding: 1px;
  line-height: 24px;
  border-bottom: 1px solid gray;
}
.jupyter-keybindings input {
  margin: 0;
  padding: 0;
  border: none;
}
.jupyter-keybindings i {
  padding: 6px;
}
.well code {
  background-color: #ffffff;
  border-color: #ababab;
  border-width: 1px;
  border-style: solid;
  padding: 2px;
  padding-top: 1px;
  padding-bottom: 1px;
}
/* CSS for the cell toolbar */
.celltoolbar {
  border: thin solid #CFCFCF;
  border-bottom: none;
  background: #EEE;
  border-radius: 2px 2px 0px 0px;
  width: 100%;
  height: 29px;
  padding-right: 4px;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
  /* Old browsers */
  -webkit-box-pack: end;
  -moz-box-pack: end;
  box-pack: end;
  /* Modern browsers */
  justify-content: flex-end;
  display: -webkit-flex;
}
@media print {
  .celltoolbar {
    display: none;
  }
}
.ctb_hideshow {
  display: none;
  vertical-align: bottom;
}
/* ctb_show is added to the ctb_hideshow div to show the cell toolbar.
   Cell toolbars are only shown when the ctb_global_show class is also set.
*/
.ctb_global_show .ctb_show.ctb_hideshow {
  display: block;
}
.ctb_global_show .ctb_show + .input_area,
.ctb_global_show .ctb_show + div.text_cell_input,
.ctb_global_show .ctb_show ~ div.text_cell_render {
  border-top-right-radius: 0px;
  border-top-left-radius: 0px;
}
.ctb_global_show .ctb_show ~ div.text_cell_render {
  border: 1px solid #cfcfcf;
}
.celltoolbar {
  font-size: 87%;
  padding-top: 3px;
}
.celltoolbar select {
  display: block;
  width: 100%;
  height: 32px;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
  background-color: #fff;
  background-image: none;
  border: 1px solid #ccc;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  -webkit-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  -o-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
  width: inherit;
  font-size: inherit;
  height: 22px;
  padding: 0px;
  display: inline-block;
}
.celltoolbar select:focus {
  border-color: #66afe9;
  outline: 0;
  -webkit-box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
  box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
}
.celltoolbar select::-moz-placeholder {
  color: #999;
  opacity: 1;
}
.celltoolbar select:-ms-input-placeholder {
  color: #999;
}
.celltoolbar select::-webkit-input-placeholder {
  color: #999;
}
.celltoolbar select::-ms-expand {
  border: 0;
  background-color: transparent;
}
.celltoolbar select[disabled],
.celltoolbar select[readonly],
fieldset[disabled] .celltoolbar select {
  background-color: #eeeeee;
  opacity: 1;
}
.celltoolbar select[disabled],
fieldset[disabled] .celltoolbar select {
  cursor: not-allowed;
}
textarea.celltoolbar select {
  height: auto;
}
select.celltoolbar select {
  height: 30px;
  line-height: 30px;
}
textarea.celltoolbar select,
select[multiple].celltoolbar select {
  height: auto;
}
.celltoolbar label {
  margin-left: 5px;
  margin-right: 5px;
}
.tags_button_container {
  width: 100%;
  display: flex;
}
.tag-container {
  display: flex;
  flex-direction: row;
  flex-grow: 1;
  overflow: hidden;
  position: relative;
}
.tag-container > * {
  margin: 0 4px;
}
.remove-tag-btn {
  margin-left: 4px;
}
.tags-input {
  display: flex;
}
.cell-tag:last-child:after {
  content: "";
  position: absolute;
  right: 0;
  width: 40px;
  height: 100%;
  /* Fade to background color of cell toolbar */
  background: linear-gradient(to right, rgba(0, 0, 0, 0), #EEE);
}
.tags-input > * {
  margin-left: 4px;
}
.cell-tag,
.tags-input input,
.tags-input button {
  display: block;
  width: 100%;
  height: 32px;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
  background-color: #fff;
  background-image: none;
  border: 1px solid #ccc;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  -webkit-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  -o-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
  box-shadow: none;
  width: inherit;
  font-size: inherit;
  height: 22px;
  line-height: 22px;
  padding: 0px 4px;
  display: inline-block;
}
.cell-tag:focus,
.tags-input input:focus,
.tags-input button:focus {
  border-color: #66afe9;
  outline: 0;
  -webkit-box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
  box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
}
.cell-tag::-moz-placeholder,
.tags-input input::-moz-placeholder,
.tags-input button::-moz-placeholder {
  color: #999;
  opacity: 1;
}
.cell-tag:-ms-input-placeholder,
.tags-input input:-ms-input-placeholder,
.tags-input button:-ms-input-placeholder {
  color: #999;
}
.cell-tag::-webkit-input-placeholder,
.tags-input input::-webkit-input-placeholder,
.tags-input button::-webkit-input-placeholder {
  color: #999;
}
.cell-tag::-ms-expand,
.tags-input input::-ms-expand,
.tags-input button::-ms-expand {
  border: 0;
  background-color: transparent;
}
.cell-tag[disabled],
.tags-input input[disabled],
.tags-input button[disabled],
.cell-tag[readonly],
.tags-input input[readonly],
.tags-input button[readonly],
fieldset[disabled] .cell-tag,
fieldset[disabled] .tags-input input,
fieldset[disabled] .tags-input button {
  background-color: #eeeeee;
  opacity: 1;
}
.cell-tag[disabled],
.tags-input input[disabled],
.tags-input button[disabled],
fieldset[disabled] .cell-tag,
fieldset[disabled] .tags-input input,
fieldset[disabled] .tags-input button {
  cursor: not-allowed;
}
textarea.cell-tag,
textarea.tags-input input,
textarea.tags-input button {
  height: auto;
}
select.cell-tag,
select.tags-input input,
select.tags-input button {
  height: 30px;
  line-height: 30px;
}
textarea.cell-tag,
textarea.tags-input input,
textarea.tags-input button,
select[multiple].cell-tag,
select[multiple].tags-input input,
select[multiple].tags-input button {
  height: auto;
}
.cell-tag,
.tags-input button {
  padding: 0px 4px;
}
.cell-tag {
  background-color: #fff;
  white-space: nowrap;
}
.tags-input input[type=text]:focus {
  outline: none;
  box-shadow: none;
  border-color: #ccc;
}
.completions {
  position: absolute;
  z-index: 110;
  overflow: hidden;
  border: 1px solid #ababab;
  border-radius: 2px;
  -webkit-box-shadow: 0px 6px 10px -1px #adadad;
  box-shadow: 0px 6px 10px -1px #adadad;
  line-height: 1;
}
.completions select {
  background: white;
  outline: none;
  border: none;
  padding: 0px;
  margin: 0px;
  overflow: auto;
  font-family: monospace;
  font-size: 110%;
  color: #000;
  width: auto;
}
.completions select option.context {
  color: #286090;
}
#kernel_logo_widget .current_kernel_logo {
  display: none;
  margin-top: -1px;
  margin-bottom: -1px;
  width: 32px;
  height: 32px;
}
[dir="rtl"] #kernel_logo_widget {
  float: left !important;
  float: left;
}
.modal .modal-body .move-path {
  display: flex;
  flex-direction: row;
  justify-content: space;
  align-items: center;
}
.modal .modal-body .move-path .server-root {
  padding-right: 20px;
}
.modal .modal-body .move-path .path-input {
  flex: 1;
}
#menubar {
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  margin-top: 1px;
}
#menubar .navbar {
  border-top: 1px;
  border-radius: 0px 0px 2px 2px;
  margin-bottom: 0px;
}
#menubar .navbar-toggle {
  float: left;
  padding-top: 7px;
  padding-bottom: 7px;
  border: none;
}
#menubar .navbar-collapse {
  clear: left;
}
[dir="rtl"] #menubar .navbar-toggle {
  float: right;
}
[dir="rtl"] #menubar .navbar-collapse {
  clear: right;
}
[dir="rtl"] #menubar .navbar-nav {
  float: right;
}
[dir="rtl"] #menubar .nav {
  padding-right: 0px;
}
[dir="rtl"] #menubar .navbar-nav > li {
  float: right;
}
[dir="rtl"] #menubar .navbar-right {
  float: left !important;
}
[dir="rtl"] ul.dropdown-menu {
  text-align: right;
  left: auto;
}
[dir="rtl"] ul#new-menu.dropdown-menu {
  right: auto;
  left: 0;
}
.nav-wrapper {
  border-bottom: 1px solid #e7e7e7;
}
i.menu-icon {
  padding-top: 4px;
}
[dir="rtl"] i.menu-icon.pull-right {
  float: left !important;
  float: left;
}
ul#help_menu li a {
  overflow: hidden;
  padding-right: 2.2em;
}
ul#help_menu li a i {
  margin-right: -1.2em;
}
[dir="rtl"] ul#help_menu li a {
  padding-left: 2.2em;
}
[dir="rtl"] ul#help_menu li a i {
  margin-right: 0;
  margin-left: -1.2em;
}
[dir="rtl"] ul#help_menu li a i.pull-right {
  float: left !important;
  float: left;
}
.dropdown-submenu {
  position: relative;
}
.dropdown-submenu > .dropdown-menu {
  top: 0;
  left: 100%;
  margin-top: -6px;
  margin-left: -1px;
}
[dir="rtl"] .dropdown-submenu > .dropdown-menu {
  right: 100%;
  margin-right: -1px;
}
.dropdown-submenu:hover > .dropdown-menu {
  display: block;
}
.dropdown-submenu > a:after {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  display: block;
  content: "\f0da";
  float: right;
  color: #333333;
  margin-top: 2px;
  margin-right: -10px;
}
.dropdown-submenu > a:after.fa-pull-left {
  margin-right: .3em;
}
.dropdown-submenu > a:after.fa-pull-right {
  margin-left: .3em;
}
.dropdown-submenu > a:after.pull-left {
  margin-right: .3em;
}
.dropdown-submenu > a:after.pull-right {
  margin-left: .3em;
}
[dir="rtl"] .dropdown-submenu > a:after {
  float: left;
  content: "\f0d9";
  margin-right: 0;
  margin-left: -10px;
}
.dropdown-submenu:hover > a:after {
  color: #262626;
}
.dropdown-submenu.pull-left {
  float: none;
}
.dropdown-submenu.pull-left > .dropdown-menu {
  left: -100%;
  margin-left: 10px;
}
#notification_area {
  float: right !important;
  float: right;
  z-index: 10;
}
[dir="rtl"] #notification_area {
  float: left !important;
  float: left;
}
.indicator_area {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
}
[dir="rtl"] .indicator_area {
  float: left !important;
  float: left;
}
#kernel_indicator {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
  border-left: 1px solid;
}
#kernel_indicator .kernel_indicator_name {
  padding-left: 5px;
  padding-right: 5px;
}
[dir="rtl"] #kernel_indicator {
  float: left !important;
  float: left;
  border-left: 0;
  border-right: 1px solid;
}
#modal_indicator {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
}
[dir="rtl"] #modal_indicator {
  float: left !important;
  float: left;
}
#readonly-indicator {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
  margin-top: 2px;
  margin-bottom: 0px;
  margin-left: 0px;
  margin-right: 0px;
  display: none;
}
.modal_indicator:before {
  width: 1.28571429em;
  text-align: center;
}
.edit_mode .modal_indicator:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f040";
}
.edit_mode .modal_indicator:before.fa-pull-left {
  margin-right: .3em;
}
.edit_mode .modal_indicator:before.fa-pull-right {
  margin-left: .3em;
}
.edit_mode .modal_indicator:before.pull-left {
  margin-right: .3em;
}
.edit_mode .modal_indicator:before.pull-right {
  margin-left: .3em;
}
.command_mode .modal_indicator:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: ' ';
}
.command_mode .modal_indicator:before.fa-pull-left {
  margin-right: .3em;
}
.command_mode .modal_indicator:before.fa-pull-right {
  margin-left: .3em;
}
.command_mode .modal_indicator:before.pull-left {
  margin-right: .3em;
}
.command_mode .modal_indicator:before.pull-right {
  margin-left: .3em;
}
.kernel_idle_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f10c";
}
.kernel_idle_icon:before.fa-pull-left {
  margin-right: .3em;
}
.kernel_idle_icon:before.fa-pull-right {
  margin-left: .3em;
}
.kernel_idle_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_idle_icon:before.pull-right {
  margin-left: .3em;
}
.kernel_busy_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f111";
}
.kernel_busy_icon:before.fa-pull-left {
  margin-right: .3em;
}
.kernel_busy_icon:before.fa-pull-right {
  margin-left: .3em;
}
.kernel_busy_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_busy_icon:before.pull-right {
  margin-left: .3em;
}
.kernel_dead_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f1e2";
}
.kernel_dead_icon:before.fa-pull-left {
  margin-right: .3em;
}
.kernel_dead_icon:before.fa-pull-right {
  margin-left: .3em;
}
.kernel_dead_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_dead_icon:before.pull-right {
  margin-left: .3em;
}
.kernel_disconnected_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f127";
}
.kernel_disconnected_icon:before.fa-pull-left {
  margin-right: .3em;
}
.kernel_disconnected_icon:before.fa-pull-right {
  margin-left: .3em;
}
.kernel_disconnected_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_disconnected_icon:before.pull-right {
  margin-left: .3em;
}
.notification_widget {
  color: #777;
  z-index: 10;
  background: rgba(240, 240, 240, 0.5);
  margin-right: 4px;
  color: #333;
  background-color: #fff;
  border-color: #ccc;
}
.notification_widget:focus,
.notification_widget.focus {
  color: #333;
  background-color: #e6e6e6;
  border-color: #8c8c8c;
}
.notification_widget:hover {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.notification_widget:active,
.notification_widget.active,
.open > .dropdown-toggle.notification_widget {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.notification_widget:active:hover,
.notification_widget.active:hover,
.open > .dropdown-toggle.notification_widget:hover,
.notification_widget:active:focus,
.notification_widget.active:focus,
.open > .dropdown-toggle.notification_widget:focus,
.notification_widget:active.focus,
.notification_widget.active.focus,
.open > .dropdown-toggle.notification_widget.focus {
  color: #333;
  background-color: #d4d4d4;
  border-color: #8c8c8c;
}
.notification_widget:active,
.notification_widget.active,
.open > .dropdown-toggle.notification_widget {
  background-image: none;
}
.notification_widget.disabled:hover,
.notification_widget[disabled]:hover,
fieldset[disabled] .notification_widget:hover,
.notification_widget.disabled:focus,
.notification_widget[disabled]:focus,
fieldset[disabled] .notification_widget:focus,
.notification_widget.disabled.focus,
.notification_widget[disabled].focus,
fieldset[disabled] .notification_widget.focus {
  background-color: #fff;
  border-color: #ccc;
}
.notification_widget .badge {
  color: #fff;
  background-color: #333;
}
.notification_widget.warning {
  color: #fff;
  background-color: #f0ad4e;
  border-color: #eea236;
}
.notification_widget.warning:focus,
.notification_widget.warning.focus {
  color: #fff;
  background-color: #ec971f;
  border-color: #985f0d;
}
.notification_widget.warning:hover {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.notification_widget.warning:active,
.notification_widget.warning.active,
.open > .dropdown-toggle.notification_widget.warning {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.notification_widget.warning:active:hover,
.notification_widget.warning.active:hover,
.open > .dropdown-toggle.notification_widget.warning:hover,
.notification_widget.warning:active:focus,
.notification_widget.warning.active:focus,
.open > .dropdown-toggle.notification_widget.warning:focus,
.notification_widget.warning:active.focus,
.notification_widget.warning.active.focus,
.open > .dropdown-toggle.notification_widget.warning.focus {
  color: #fff;
  background-color: #d58512;
  border-color: #985f0d;
}
.notification_widget.warning:active,
.notification_widget.warning.active,
.open > .dropdown-toggle.notification_widget.warning {
  background-image: none;
}
.notification_widget.warning.disabled:hover,
.notification_widget.warning[disabled]:hover,
fieldset[disabled] .notification_widget.warning:hover,
.notification_widget.warning.disabled:focus,
.notification_widget.warning[disabled]:focus,
fieldset[disabled] .notification_widget.warning:focus,
.notification_widget.warning.disabled.focus,
.notification_widget.warning[disabled].focus,
fieldset[disabled] .notification_widget.warning.focus {
  background-color: #f0ad4e;
  border-color: #eea236;
}
.notification_widget.warning .badge {
  color: #f0ad4e;
  background-color: #fff;
}
.notification_widget.success {
  color: #fff;
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.notification_widget.success:focus,
.notification_widget.success.focus {
  color: #fff;
  background-color: #449d44;
  border-color: #255625;
}
.notification_widget.success:hover {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.notification_widget.success:active,
.notification_widget.success.active,
.open > .dropdown-toggle.notification_widget.success {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.notification_widget.success:active:hover,
.notification_widget.success.active:hover,
.open > .dropdown-toggle.notification_widget.success:hover,
.notification_widget.success:active:focus,
.notification_widget.success.active:focus,
.open > .dropdown-toggle.notification_widget.success:focus,
.notification_widget.success:active.focus,
.notification_widget.success.active.focus,
.open > .dropdown-toggle.notification_widget.success.focus {
  color: #fff;
  background-color: #398439;
  border-color: #255625;
}
.notification_widget.success:active,
.notification_widget.success.active,
.open > .dropdown-toggle.notification_widget.success {
  background-image: none;
}
.notification_widget.success.disabled:hover,
.notification_widget.success[disabled]:hover,
fieldset[disabled] .notification_widget.success:hover,
.notification_widget.success.disabled:focus,
.notification_widget.success[disabled]:focus,
fieldset[disabled] .notification_widget.success:focus,
.notification_widget.success.disabled.focus,
.notification_widget.success[disabled].focus,
fieldset[disabled] .notification_widget.success.focus {
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.notification_widget.success .badge {
  color: #5cb85c;
  background-color: #fff;
}
.notification_widget.info {
  color: #fff;
  background-color: #5bc0de;
  border-color: #46b8da;
}
.notification_widget.info:focus,
.notification_widget.info.focus {
  color: #fff;
  background-color: #31b0d5;
  border-color: #1b6d85;
}
.notification_widget.info:hover {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.notification_widget.info:active,
.notification_widget.info.active,
.open > .dropdown-toggle.notification_widget.info {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.notification_widget.info:active:hover,
.notification_widget.info.active:hover,
.open > .dropdown-toggle.notification_widget.info:hover,
.notification_widget.info:active:focus,
.notification_widget.info.active:focus,
.open > .dropdown-toggle.notification_widget.info:focus,
.notification_widget.info:active.focus,
.notification_widget.info.active.focus,
.open > .dropdown-toggle.notification_widget.info.focus {
  color: #fff;
  background-color: #269abc;
  border-color: #1b6d85;
}
.notification_widget.info:active,
.notification_widget.info.active,
.open > .dropdown-toggle.notification_widget.info {
  background-image: none;
}
.notification_widget.info.disabled:hover,
.notification_widget.info[disabled]:hover,
fieldset[disabled] .notification_widget.info:hover,
.notification_widget.info.disabled:focus,
.notification_widget.info[disabled]:focus,
fieldset[disabled] .notification_widget.info:focus,
.notification_widget.info.disabled.focus,
.notification_widget.info[disabled].focus,
fieldset[disabled] .notification_widget.info.focus {
  background-color: #5bc0de;
  border-color: #46b8da;
}
.notification_widget.info .badge {
  color: #5bc0de;
  background-color: #fff;
}
.notification_widget.danger {
  color: #fff;
  background-color: #d9534f;
  border-color: #d43f3a;
}
.notification_widget.danger:focus,
.notification_widget.danger.focus {
  color: #fff;
  background-color: #c9302c;
  border-color: #761c19;
}
.notification_widget.danger:hover {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.notification_widget.danger:active,
.notification_widget.danger.active,
.open > .dropdown-toggle.notification_widget.danger {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.notification_widget.danger:active:hover,
.notification_widget.danger.active:hover,
.open > .dropdown-toggle.notification_widget.danger:hover,
.notification_widget.danger:active:focus,
.notification_widget.danger.active:focus,
.open > .dropdown-toggle.notification_widget.danger:focus,
.notification_widget.danger:active.focus,
.notification_widget.danger.active.focus,
.open > .dropdown-toggle.notification_widget.danger.focus {
  color: #fff;
  background-color: #ac2925;
  border-color: #761c19;
}
.notification_widget.danger:active,
.notification_widget.danger.active,
.open > .dropdown-toggle.notification_widget.danger {
  background-image: none;
}
.notification_widget.danger.disabled:hover,
.notification_widget.danger[disabled]:hover,
fieldset[disabled] .notification_widget.danger:hover,
.notification_widget.danger.disabled:focus,
.notification_widget.danger[disabled]:focus,
fieldset[disabled] .notification_widget.danger:focus,
.notification_widget.danger.disabled.focus,
.notification_widget.danger[disabled].focus,
fieldset[disabled] .notification_widget.danger.focus {
  background-color: #d9534f;
  border-color: #d43f3a;
}
.notification_widget.danger .badge {
  color: #d9534f;
  background-color: #fff;
}
div#pager {
  background-color: #fff;
  font-size: 14px;
  line-height: 20px;
  overflow: hidden;
  display: none;
  position: fixed;
  bottom: 0px;
  width: 100%;
  max-height: 50%;
  padding-top: 8px;
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  /* Display over codemirror */
  z-index: 100;
  /* Hack which prevents jquery ui resizable from changing top. */
  top: auto !important;
}
div#pager pre {
  line-height: 1.21429em;
  color: #000;
  background-color: #f7f7f7;
  padding: 0.4em;
}
div#pager #pager-button-area {
  position: absolute;
  top: 8px;
  right: 20px;
}
div#pager #pager-contents {
  position: relative;
  overflow: auto;
  width: 100%;
  height: 100%;
}
div#pager #pager-contents #pager-container {
  position: relative;
  padding: 15px 0px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
div#pager .ui-resizable-handle {
  top: 0px;
  height: 8px;
  background: #f7f7f7;
  border-top: 1px solid #cfcfcf;
  border-bottom: 1px solid #cfcfcf;
  /* This injects handle bars (a short, wide = symbol) for 
        the resize handle. */
}
div#pager .ui-resizable-handle::after {
  content: '';
  top: 2px;
  left: 50%;
  height: 3px;
  width: 30px;
  margin-left: -15px;
  position: absolute;
  border-top: 1px solid #cfcfcf;
}
.quickhelp {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
  line-height: 1.8em;
}
.shortcut_key {
  display: inline-block;
  width: 21ex;
  text-align: right;
  font-family: monospace;
}
.shortcut_descr {
  display: inline-block;
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
span.save_widget {
  height: 30px;
  margin-top: 4px;
  display: flex;
  justify-content: flex-start;
  align-items: baseline;
  width: 50%;
  flex: 1;
}
span.save_widget span.filename {
  height: 100%;
  line-height: 1em;
  margin-left: 16px;
  border: none;
  font-size: 146.5%;
  text-overflow: ellipsis;
  overflow: hidden;
  white-space: nowrap;
  border-radius: 2px;
}
span.save_widget span.filename:hover {
  background-color: #e6e6e6;
}
[dir="rtl"] span.save_widget.pull-left {
  float: right !important;
  float: right;
}
[dir="rtl"] span.save_widget span.filename {
  margin-left: 0;
  margin-right: 16px;
}
span.checkpoint_status,
span.autosave_status {
  font-size: small;
  white-space: nowrap;
  padding: 0 5px;
}
@media (max-width: 767px) {
  span.save_widget {
    font-size: small;
    padding: 0 0 0 5px;
  }
  span.checkpoint_status,
  span.autosave_status {
    display: none;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  span.checkpoint_status {
    display: none;
  }
  span.autosave_status {
    font-size: x-small;
  }
}
.toolbar {
  padding: 0px;
  margin-left: -5px;
  margin-top: 2px;
  margin-bottom: 5px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
.toolbar select,
.toolbar label {
  width: auto;
  vertical-align: middle;
  margin-right: 2px;
  margin-bottom: 0px;
  display: inline;
  font-size: 92%;
  margin-left: 0.3em;
  margin-right: 0.3em;
  padding: 0px;
  padding-top: 3px;
}
.toolbar .btn {
  padding: 2px 8px;
}
.toolbar .btn-group {
  margin-top: 0px;
  margin-left: 5px;
}
.toolbar-btn-label {
  margin-left: 6px;
}
#maintoolbar {
  margin-bottom: -3px;
  margin-top: -8px;
  border: 0px;
  min-height: 27px;
  margin-left: 0px;
  padding-top: 11px;
  padding-bottom: 3px;
}
#maintoolbar .navbar-text {
  float: none;
  vertical-align: middle;
  text-align: right;
  margin-left: 5px;
  margin-right: 0px;
  margin-top: 0px;
}
.select-xs {
  height: 24px;
}
[dir="rtl"] .btn-group > .btn,
.btn-group-vertical > .btn {
  float: right;
}
.pulse,
.dropdown-menu > li > a.pulse,
li.pulse > a.dropdown-toggle,
li.pulse.open > a.dropdown-toggle {
  background-color: #F37626;
  color: white;
}
/**
 * Primary styles
 *
 * Author: Jupyter Development Team
 */
/** WARNING IF YOU ARE EDITTING THIS FILE, if this is a .css file, It has a lot
 * of chance of beeing generated from the ../less/[samename].less file, you can
 * try to get back the less file by reverting somme commit in history
 **/
/*
 * We'll try to get something pretty, so we
 * have some strange css to have the scroll bar on
 * the left with fix button on the top right of the tooltip
 */
@-moz-keyframes fadeOut {
  from {
    opacity: 1;
  }
  to {
    opacity: 0;
  }
}
@-webkit-keyframes fadeOut {
  from {
    opacity: 1;
  }
  to {
    opacity: 0;
  }
}
@-moz-keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}
@-webkit-keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}
/*properties of tooltip after "expand"*/
.bigtooltip {
  overflow: auto;
  height: 200px;
  -webkit-transition-property: height;
  -webkit-transition-duration: 500ms;
  -moz-transition-property: height;
  -moz-transition-duration: 500ms;
  transition-property: height;
  transition-duration: 500ms;
}
/*properties of tooltip before "expand"*/
.smalltooltip {
  -webkit-transition-property: height;
  -webkit-transition-duration: 500ms;
  -moz-transition-property: height;
  -moz-transition-duration: 500ms;
  transition-property: height;
  transition-duration: 500ms;
  text-overflow: ellipsis;
  overflow: hidden;
  height: 80px;
}
.tooltipbuttons {
  position: absolute;
  padding-right: 15px;
  top: 0px;
  right: 0px;
}
.tooltiptext {
  /*avoid the button to overlap on some docstring*/
  padding-right: 30px;
}
.ipython_tooltip {
  max-width: 700px;
  /*fade-in animation when inserted*/
  -webkit-animation: fadeOut 400ms;
  -moz-animation: fadeOut 400ms;
  animation: fadeOut 400ms;
  -webkit-animation: fadeIn 400ms;
  -moz-animation: fadeIn 400ms;
  animation: fadeIn 400ms;
  vertical-align: middle;
  background-color: #f7f7f7;
  overflow: visible;
  border: #ababab 1px solid;
  outline: none;
  padding: 3px;
  margin: 0px;
  padding-left: 7px;
  font-family: monospace;
  min-height: 50px;
  -moz-box-shadow: 0px 6px 10px -1px #adadad;
  -webkit-box-shadow: 0px 6px 10px -1px #adadad;
  box-shadow: 0px 6px 10px -1px #adadad;
  border-radius: 2px;
  position: absolute;
  z-index: 1000;
}
.ipython_tooltip a {
  float: right;
}
.ipython_tooltip .tooltiptext pre {
  border: 0;
  border-radius: 0;
  font-size: 100%;
  background-color: #f7f7f7;
}
.pretooltiparrow {
  left: 0px;
  margin: 0px;
  top: -16px;
  width: 40px;
  height: 16px;
  overflow: hidden;
  position: absolute;
}
.pretooltiparrow:before {
  background-color: #f7f7f7;
  border: 1px #ababab solid;
  z-index: 11;
  content: "";
  position: absolute;
  left: 15px;
  top: 10px;
  width: 25px;
  height: 25px;
  -webkit-transform: rotate(45deg);
  -moz-transform: rotate(45deg);
  -ms-transform: rotate(45deg);
  -o-transform: rotate(45deg);
}
ul.typeahead-list i {
  margin-left: -10px;
  width: 18px;
}
[dir="rtl"] ul.typeahead-list i {
  margin-left: 0;
  margin-right: -10px;
}
ul.typeahead-list {
  max-height: 80vh;
  overflow: auto;
}
ul.typeahead-list > li > a {
  /** Firefox bug **/
  /* see https://github.com/jupyter/notebook/issues/559 */
  white-space: normal;
}
ul.typeahead-list  > li > a.pull-right {
  float: left !important;
  float: left;
}
[dir="rtl"] .typeahead-list {
  text-align: right;
}
.cmd-palette .modal-body {
  padding: 7px;
}
.cmd-palette form {
  background: white;
}
.cmd-palette input {
  outline: none;
}
.no-shortcut {
  min-width: 20px;
  color: transparent;
}
[dir="rtl"] .no-shortcut.pull-right {
  float: left !important;
  float: left;
}
[dir="rtl"] .command-shortcut.pull-right {
  float: left !important;
  float: left;
}
.command-shortcut:before {
  content: "(command mode)";
  padding-right: 3px;
  color: #777777;
}
.edit-shortcut:before {
  content: "(edit)";
  padding-right: 3px;
  color: #777777;
}
[dir="rtl"] .edit-shortcut.pull-right {
  float: left !important;
  float: left;
}
#find-and-replace #replace-preview .match,
#find-and-replace #replace-preview .insert {
  background-color: #BBDEFB;
  border-color: #90CAF9;
  border-style: solid;
  border-width: 1px;
  border-radius: 0px;
}
[dir="ltr"] #find-and-replace .input-group-btn + .form-control {
  border-left: none;
}
[dir="rtl"] #find-and-replace .input-group-btn + .form-control {
  border-right: none;
}
#find-and-replace #replace-preview .replace .match {
  background-color: #FFCDD2;
  border-color: #EF9A9A;
  border-radius: 0px;
}
#find-and-replace #replace-preview .replace .insert {
  background-color: #C8E6C9;
  border-color: #A5D6A7;
  border-radius: 0px;
}
#find-and-replace #replace-preview {
  max-height: 60vh;
  overflow: auto;
}
#find-and-replace #replace-preview pre {
  padding: 5px 10px;
}
.terminal-app {
  background: #EEE;
}
.terminal-app #header {
  background: #fff;
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
}
.terminal-app .terminal {
  width: 100%;
  float: left;
  font-family: monospace;
  color: white;
  background: black;
  padding: 0.4em;
  border-radius: 2px;
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.4);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.4);
}
.terminal-app .terminal,
.terminal-app .terminal dummy-screen {
  line-height: 1em;
  font-size: 14px;
}
.terminal-app .terminal .xterm-rows {
  padding: 10px;
}
.terminal-app .terminal-cursor {
  color: black;
  background: white;
}
.terminal-app #terminado-container {
  margin-top: 20px;
}
/*# sourceMappingURL=style.min.css.map */
    </style>
<style type="text/css">
    .highlight .hll { background-color: #ffffcc }
.highlight  { background: #f8f8f8; }
.highlight .c { color: #408080; font-style: italic } /* Comment */
.highlight .err { border: 1px solid #FF0000 } /* Error */
.highlight .k { color: #008000; font-weight: bold } /* Keyword */
.highlight .o { color: #666666 } /* Operator */
.highlight .ch { color: #408080; font-style: italic } /* Comment.Hashbang */
.highlight .cm { color: #408080; font-style: italic } /* Comment.Multiline */
.highlight .cp { color: #BC7A00 } /* Comment.Preproc */
.highlight .cpf { color: #408080; font-style: italic } /* Comment.PreprocFile */
.highlight .c1 { color: #408080; font-style: italic } /* Comment.Single */
.highlight .cs { color: #408080; font-style: italic } /* Comment.Special */
.highlight .gd { color: #A00000 } /* Generic.Deleted */
.highlight .ge { font-style: italic } /* Generic.Emph */
.highlight .gr { color: #FF0000 } /* Generic.Error */
.highlight .gh { color: #000080; font-weight: bold } /* Generic.Heading */
.highlight .gi { color: #00A000 } /* Generic.Inserted */
.highlight .go { color: #888888 } /* Generic.Output */
.highlight .gp { color: #000080; font-weight: bold } /* Generic.Prompt */
.highlight .gs { font-weight: bold } /* Generic.Strong */
.highlight .gu { color: #800080; font-weight: bold } /* Generic.Subheading */
.highlight .gt { color: #0044DD } /* Generic.Traceback */
.highlight .kc { color: #008000; font-weight: bold } /* Keyword.Constant */
.highlight .kd { color: #008000; font-weight: bold } /* Keyword.Declaration */
.highlight .kn { color: #008000; font-weight: bold } /* Keyword.Namespace */
.highlight .kp { color: #008000 } /* Keyword.Pseudo */
.highlight .kr { color: #008000; font-weight: bold } /* Keyword.Reserved */
.highlight .kt { color: #B00040 } /* Keyword.Type */
.highlight .m { color: #666666 } /* Literal.Number */
.highlight .s { color: #BA2121 } /* Literal.String */
.highlight .na { color: #7D9029 } /* Name.Attribute */
.highlight .nb { color: #008000 } /* Name.Builtin */
.highlight .nc { color: #0000FF; font-weight: bold } /* Name.Class */
.highlight .no { color: #880000 } /* Name.Constant */
.highlight .nd { color: #AA22FF } /* Name.Decorator */
.highlight .ni { color: #999999; font-weight: bold } /* Name.Entity */
.highlight .ne { color: #D2413A; font-weight: bold } /* Name.Exception */
.highlight .nf { color: #0000FF } /* Name.Function */
.highlight .nl { color: #A0A000 } /* Name.Label */
.highlight .nn { color: #0000FF; font-weight: bold } /* Name.Namespace */
.highlight .nt { color: #008000; font-weight: bold } /* Name.Tag */
.highlight .nv { color: #19177C } /* Name.Variable */
.highlight .ow { color: #AA22FF; font-weight: bold } /* Operator.Word */
.highlight .w { color: #bbbbbb } /* Text.Whitespace */
.highlight .mb { color: #666666 } /* Literal.Number.Bin */
.highlight .mf { color: #666666 } /* Literal.Number.Float */
.highlight .mh { color: #666666 } /* Literal.Number.Hex */
.highlight .mi { color: #666666 } /* Literal.Number.Integer */
.highlight .mo { color: #666666 } /* Literal.Number.Oct */
.highlight .sa { color: #BA2121 } /* Literal.String.Affix */
.highlight .sb { color: #BA2121 } /* Literal.String.Backtick */
.highlight .sc { color: #BA2121 } /* Literal.String.Char */
.highlight .dl { color: #BA2121 } /* Literal.String.Delimiter */
.highlight .sd { color: #BA2121; font-style: italic } /* Literal.String.Doc */
.highlight .s2 { color: #BA2121 } /* Literal.String.Double */
.highlight .se { color: #BB6622; font-weight: bold } /* Literal.String.Escape */
.highlight .sh { color: #BA2121 } /* Literal.String.Heredoc */
.highlight .si { color: #BB6688; font-weight: bold } /* Literal.String.Interpol */
.highlight .sx { color: #008000 } /* Literal.String.Other */
.highlight .sr { color: #BB6688 } /* Literal.String.Regex */
.highlight .s1 { color: #BA2121 } /* Literal.String.Single */
.highlight .ss { color: #19177C } /* Literal.String.Symbol */
.highlight .bp { color: #008000 } /* Name.Builtin.Pseudo */
.highlight .fm { color: #0000FF } /* Name.Function.Magic */
.highlight .vc { color: #19177C } /* Name.Variable.Class */
.highlight .vg { color: #19177C } /* Name.Variable.Global */
.highlight .vi { color: #19177C } /* Name.Variable.Instance */
.highlight .vm { color: #19177C } /* Name.Variable.Magic */
.highlight .il { color: #666666 } /* Literal.Number.Integer.Long */
    </style>


<style type="text/css">
/* Overrides of notebook CSS for static HTML export */
body {
  overflow: visible;
  padding: 8px;
}

div#notebook {
  overflow: visible;
  border-top: none;
}div#notebook-container{
  padding: 6ex 12ex 8ex 12ex;
}@media print {
  div.cell {
    display: block;
    page-break-inside: avoid;
  } 
  div.output_wrapper { 
    display: block;
    page-break-inside: avoid; 
  }
  div.output { 
    display: block;
    page-break-inside: avoid; 
  }
}
</style>

<!-- Custom stylesheet, it must be in the same directory as the html file -->
<link rel="stylesheet" href="custom.css">

<!-- Loading mathjax macro -->
<!-- Load mathjax -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=TeX-AMS_HTML"></script>
    <!-- MathJax configuration -->
    <script type="text/x-mathjax-config">
    MathJax.Hub.Config({
        tex2jax: {
            inlineMath: [ ['$','$'], ["\\(","\\)"] ],
            displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
            processEscapes: true,
            processEnvironments: true
        },
        // Center justify equations in code and markdown cells. Elsewhere
        // we use CSS to left justify single line equations in code cells.
        displayAlign: 'center',
        "HTML-CSS": {
            styles: {'.MathJax_Display': {"margin": 0}},
            linebreaks: { automatic: true }
        }
    });
    </script>
    <!-- End of mathjax configuration --></head>
<body>
  <div tabindex="-1" id="notebook" class="border-box-sizing">
    <div class="container" id="notebook-container">

<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<hr>
<p>layout: post
title: Exploration in Modeling with SF 311
subtitle: Trying to predict 
cover-img: /assets/img/rainbow.jpg</p>
<h2 id="tags:-[SF311,-opendata,-public,-bay-area,-quality-of-life,-modeling,predictions]">tags: [SF311, opendata, public, bay area, quality of life, modeling,predictions]<a class="anchor-link" href="#tags:-[SF311,-opendata,-public,-bay-area,-quality-of-life,-modeling,predictions]">&#182;</a></h2>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Dataset-Introduction:-San-Francisco-311-Cases">Dataset Introduction: San Francisco 311 Cases<a class="anchor-link" href="#Dataset-Introduction:-San-Francisco-311-Cases">&#182;</a></h2><p>Provided by <a href="https://data.sfgov.org/City-Infrastructure/311-Cases/vw6y-z8j6">DataSF</a>, this set is a monstrosity containing about 4.25 Million rows and counting. For those not familiar, 311 is a general customer service number for city government, most commonly associated with non-emergency complaints. 311 cases can be created via phone, mobile, and web. The dataset covers the time period from July 2008 until present.</p>
<p>In order to do data exploration and analysis of this dataset we needed to make a working sample to reduce memory and cpu usage upfront: 
    <pre><code>awk 'BEGIN {srand()} !/^$/ { if (rand() <= .01 || FNR==1) print $0}' filename</code></pre></p>
<p>Further information about the dataset can be found <a href="https://support.datasf.org/help/311-case-data-faq">here.</a></p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="One-of-the-fascinating-aspects-of-city-living-is-human-behavior.">One of the fascinating aspects of city living is human behavior.<a class="anchor-link" href="#One-of-the-fascinating-aspects-of-city-living-is-human-behavior.">&#182;</a></h2><p>One behavior every human seems to enjoy is complaining, in some shape way or form. Whether its done in exercise of right to recourse, for entertainment, or out of civic duty, we encounter a situtation that make us uncomfortable... in SF, we file a 311 case on it. You can even do it via Twitter! I wanted to focus on the subset of cases that reflect the spirit of 'complaints--' in particular those concerned with the behavior of others-- and generally require some sort of short term physical response.</p>
<p>Sadly, it seems there aren't many cases filed to commend folks for good behavior, so we will be looking at mostly negative or unpleasant situations here. Accordingly, we have attempted to exclude cases concerning general administrative requests, such as building permits, tree maintenance,  and the like. In addition, despite it being filled with negative comments, I also chose to exclude the muni category, insofar as the Muni (city bus &amp; train operators) is its own organization with its own culture, that I don't care to upset by pointout the exceedingly high volume of complaints.</p>
<p>From my personal observation, corroborated by many of my peers, once the 311 case is filed it goes into a black box, and we can only hope to guess at if or when the matter will be addressed. This can be very frustrating for the complainant, and in turn likely results in corresponding frustration for the people inside the black box, who receive many repeated complaints each day
.</p>
<h3 id="If-only-there-were-a-way-to-predict-how-long-it-would-take-to-resolve-each-issue...">If only there were a way to <strong><em>predict</em></strong> <em>how long it would take to resolve each issue...</em><a class="anchor-link" href="#If-only-there-were-a-way-to-predict-how-long-it-would-take-to-resolve-each-issue...">&#182;</a></h3><p>Well, luckily, there <em>are</em> ways, in particular statistical learning models, and we shall see what fruit they may bear.</p>
<h3 id="Datacleaning-highlights:">Datacleaning highlights:<a class="anchor-link" href="#Datacleaning-highlights:">&#182;</a></h3><ul>
<li>encode, convert or cast every single feature</li>
<li>extreme values</li>
<li>malformed data</li>
<li>datetime index</li>
<li>4.2M rows</li>
</ul>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">

</div>
<div class="cell border-box-sizing code_cell rendered">

</div>
<div class="cell border-box-sizing code_cell rendered">

</div>
<div class="cell border-box-sizing code_cell rendered">

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="MUNI---A-short-Detour">MUNI - A short Detour<a class="anchor-link" href="#MUNI---A-short-Detour">&#182;</a></h3><p>Here are the top cases filed against the city's public tranportation operation:</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">

<div class="output_wrapper">
<div class="output">

<div class="output_area">



<div class="output_text output_subarea output_execute_result">
<pre>MUNI - Conduct_Inattentiveness_Negligence                         0.240359
MUNI - Services_Service_Delivery_Facilities                       0.160987
MUNI - Conduct_Discourteous_Insensitive_Inappropriate_Conduct     0.112108
MUNI - Conduct_Unsafe_Operation                                   0.078027
MUNI - Services_Miscellaneous                                     0.055157
MUNI  - Services_Service_Delivery_Facilities                      0.051121
MUNI - Services_Service_Planning                                  0.046188
MUNI  -                                                           0.045291
MUNI  - Conduct_Discourteous_Insensitive_Inappropriate_Conduct    0.039462
MUNI - Commendation                                               0.039013
MUNI  - Conduct_Inattentiveness_Negligence                        0.035426
MUNI  - Services_Miscellaneous                                    0.031839
MUNI  - Conduct_Unsafe_Operation                                  0.024664
MUNI  - Services_Service_Planning                                 0.021525
MUNI - Services_Criminal_Activity                                 0.016592
MUNI  - Services_Criminal_Activity                                0.001794
SSP SFMTA Feedback                                                0.000448
Name: Request Type, dtype: float64</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="So....-I-was-wrong-about-it-being-ALL-BAD...">So.... I was wrong about it being ALL BAD...<a class="anchor-link" href="#So....-I-was-wrong-about-it-being-ALL-BAD...">&#182;</a></h3><p>Fully <strong>3%</strong> of the MUNI cases are commendations for MUNI employees. Sounds about right. I would draw your attention to the subject of the remaining cases but I promised not to bash them...</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="We-want-to-predict-how-long-it-will-take-for-our-ticket-to-be-resolved-but-that-feature-didnt't-exist.">We want to predict how long it will take for our ticket to be resolved but that feature didnt't exist.<a class="anchor-link" href="#We-want-to-predict-how-long-it-will-take-for-our-ticket-to-be-resolved-but-that-feature-didnt't-exist.">&#182;</a></h3><p>So it was created by finding the time difference between the <em>Opened</em> and <em>Closed</em> timestamp. We will call this the <strong><em>Time To Resolution</em></strong> or <strong>*ttr</strong> for short:</p>

<pre><code>  wrangler.make_feature('ttr',['Opened','Closed'],lambda x : x[1]-x[0])

</code></pre>
<p>In order to extract more predictive power from the past data, we create another feature, called <strong><em>workload</em></strong>, which is simply the number of cases that were open at the time of case creation.</p>

<pre><code>def calc_open_cases(sometime):  #input time
    df= wrangler.working[['CaseID','Opened','Closed']]
    opened_prior = df['Opened'] &lt; sometime        # cases opened before it,
    not_closed_prior = ~(df['Closed'] &lt; sometime) # not closed, 
    open_at_thattime = opened_prior &amp; not_closed_prior  #and 
    return open_at_thattime.sum()
</code></pre>
<h3 id="Now-lets-look-more-at-these-and-our-features:">Now lets look more at these and our features:<a class="anchor-link" href="#Now-lets-look-more-at-these-and-our-features:">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">

<div class="output_wrapper">
<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">
<pre>dropping all in droprows..

applying mask:  drop_missing_caseid

applying mask:  drop null open date

applying mask:  muni

applying mask:  svc_req

applying mask:  gen_req

 dropping (0,) rows

working df is being changed..
converting Opened to datetime
converting Closed to datetime
converting Updated to datetime

 Encoding, changing working copy..
  ... encoding column:  Request Details
  ... encoding column:  Neighborhood
  ... encoding column:  Police District
  ... encoding column:  Analysis Neighborhoods
  ... encoding column:  Neighborhoods
  ... encoding column:  Street
  ... encoding column:  Media URL
  ... encoding column:  CaseID

added feature: ttr
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">

<div class="output_wrapper">
<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">
<pre>
added feature: ttr
</pre>
</div>
</div>

<div class="output_area">

<div class="output_subarea output_stream output_stderr output_text">
<pre>/home/dliu/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:17: UserWarning:

Boolean Series key will be reindexed to match DataFrame index.

</pre>
</div>
</div>

<div class="output_area">



<div class="output_text output_subarea output_execute_result">
<pre>(0, 25)</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="DATA-Exploration:">DATA Exploration:<a class="anchor-link" href="#DATA-Exploration:">&#182;</a></h1><h2 id="Daily-Case-Statistics">Daily Case Statistics<a class="anchor-link" href="#Daily-Case-Statistics">&#182;</a></h2><ul>
<li>upward trend in new cases, workload</li>
<li>non-monotonic increase</li>
<li>extreme values in TTR</li>
</ul>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">

<div class="output_wrapper">
<div class="output">

<div class="output_area">



<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABI0AAAIBCAYAAADEak+tAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+j8jraAAAgAElEQVR4nOydd5gUVfb3v9U9DEMGQdFZlDWsuq4oCIgigrv6irv7E10Mq7u45rgoGEAMBAFRBBlQFAQJKiISBQGBJaPkPOTMMAxhmJxD3/P+UR2qq291V3dXh5k5n+eZZ7qr7j331K1b1XVPnXOuQkQEhmEYhmEYhmEYhmEYhtFgi7UCDMMwDMMwDMMwDMMwTPzBRiOGYRiGYRiGYRiGYRjGBzYaMQzDMAzDMAzDMAzDMD6w0YhhGIZhGIZhGIZhGIbxgY1GDMMwDMMwDMMwDMMwjA9sNGIYhmEYhmEYhmEYhmF8YKMRwzAMwzBxRb9+/fDiiy/GWg0mCPicMQzDMEz1JCHWCjAMwzAME9+8//77mD9/PgCAiFBRUYFatWpBURQAwMsvv4xXXnnFsH56ejp27NiB+++/Pyr6MgzDMAzDMNbARiOGYRiGYfwydOhQDB06FACQmpqKhx9+GEuWLEGLFi1M1V+2bBk2bdrERiOGYRiGYZgqBoenMQzDMAwTFuXl5RgxYgTuvvtu3HTTTejWrRtWrFgBAPjyyy8xYsQIrF27Fq1atcK5c+dQXl6OIUOGoHPnzmjTpg26deuGNWvWmGpr7ty56Nq1K5YtW4auXbuidevWeOKJJ3Du3Dl3mZUrV+Lhhx9GmzZt0KlTJ4wYMQIOhwMbNmxA69atUVlZ6db75ptvxjvvvOOuu3z5cnTq1AlE5NN2RkYGXnrpJdxyyy3o1KkThg0b5paVn5+Pt956C3fccQfatGmDf/7zn9i1a5e77u7du/H444+jbdu2aN++PV544QWcOXPGvX/27Nm4//770bp1a/zlL3/BpEmT3Puys7Px2muvoUOHDmjTpg0eeughbNy4Udo/mzZtwnXXXYc1a9bgvvvuw0033YTHH38cZ8+edZfZsWMHevTogXbt2qFDhw7o378/iouLverPnTsX7du3x6JFi6TtTJw4EV26dEG7du0waNAgOBwOr/3Tp09H165d0aZNG/z5z3/GxIkTAQCnTp3C9ddf79U3ANCzZ0/069dP2hbDMAzDMLGDjUYMwzAMw4TFZ599huXLl2PChAnYunUrHn30UfTq1QtpaWl45ZVX8MADD6Bz585ITU1F8+bNMXnyZKxZswZz5szB1q1b8eCDD6J3794oKCgw1d758+exatUqzJ49G0uXLsXp06cxefJkAMD+/fvRu3dvPP/889i6dSumTp2KpUuX4ptvvkHbtm1BRNi3bx8A1ZDTokULbN261S178+bNuP32292hd1r++9//omnTpli7di1+/PFHrFixwm3cGTFiBNLT07FkyRJs2rQJrVq1wmuvveau26dPH3To0AGbNm3CqlWr0LhxYwwfPhwAsHr1agwbNgwDBgzAtm3bMGrUKHz11VdYunQpACAlJQVFRUVYsWIFtmzZgn/84x/o06eP22Al44cffsC0adOwdu1aJCYmom/fvu6+e+6553Dfffdhw4YNmDdvHg4cOIBPP/3Uq/7WrVuxevVq/O1vf/ORvWHDBowePRrDhg3D+vXrceONN+J///ufe//27dsxZMgQDB8+HDt27MDIkSMxevRobNiwAZdffjluvfVW/PTTT+7yRUVFWLt2LR588EHD42EYhmEYJjaw0YhhGIZhmLCYOXMmnnvuOVx99dVITExEjx490Lx5c7fRQ89zzz2Hn376CRdffDHsdjv+/ve/o7i4GEePHjXVXnFxMXr37o0GDRqgefPm6NChg7vunDlz0KFDB3Tt2hV2ux3XXHMNnnzyScydOxeJiYlo27Yttm3bBkD1qrnnnntQWlrq9lTasmUL7rjjDp829+3bh3379qFnz56oX78+fve73yElJQXt2rUDAPTv3x+TJk1CgwYNkJiYiL/97W84e/YsMjMzAaieSHXr1kVCQgLq16+Pjz/+GKNHjwYAzJgxA926dUP79u1ht9vRunVrdO/eHXPnznXXrVWrFpKSkpCQkIAePXpg7dq1SEgwzjLw1FNPoVmzZmjcuDGeeeYZbN68Gfn5+Vi0aBGaN2+OHj16oFatWkhOTsbLL7/sbsvFQw89hHr16kmNZ0uWLEH79u1xxx13IDExEQ8//DBatmzp3t+mTRts2rQJrVu3BgC0bdsWLVq0QGpqqlv24sWLUV5eDkA1mjVr1gwdOnTwe94ZhmEYhok+nNOIYRiGYZiQycvLQ15eHq6++mqv7S1b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BEEQBEEQBGEEGY0IgiAIwgIffvghunXr5pDtDx06hLCwMGRmZgIAQkJCsHXr1hLvuyw4ePAgQkJCkJGRobq+rNtsa/2iKOK5557DokWLyqxNjwrnz59Hly5dEB4ejuvXr5d3c4hSUNpxR8u8efMwduxYO7SIIAiCeBIhoxFBEARRoUlMTERoaCjCwsIQFhaGyMhI9OzZExs2bCjvppWImJgYnDhxAj4+PuXdlMeGVatWIScnB+PGjQMg9ZlmzZohKyvLqOz169fRpEkTDBkypMzak5+fj2effVb3e9y4cbhz545d6v7666/BGMOhQ4fg7+9vlzqJis2UKVNw4cIFfP755+XdFIIgCKICQkYjgiAIosIzadIknDhxAidOnMDBgwcxbtw4vP7669iyZUt5N40oZ3JycrBy5Uq8+OKL0Gg0uuXe3t744YcfjMp/99138PX1LdM2nT59Go0aNdL9vn37Nvz8/OxS9927dxEYGAgXFxe71EdUfFxcXPDCCy9gyZIlePjwYXk3hyAIgqhgkNGIIAiCeKxwcXFBcnIykpOTFSFM69evR/fu3REREYHExESsXr1at45zjg8//BCJiYmIjIxEx44d8e2335rcx549exAVFYUjR46ort+4cSOeeuopREZGYsKECXjw4IFi/datW9GjRw9ERkaiTZs2eOutt1BUVATAdOjX4sWL0aVLF8WyzMxMhIaG4uDBg4rl69evR8eOHXW/b9y4gZCQECxevFi37JNPPkFqaioA4MKFCxgxYgRiY2PRokULvPjii0hPTwcAXLlyBSEhIfjqq6/QunVrrFy50uh4r127hjZt2uCTTz4xWpefn48FCxYgKSkJzZs3R48ePfC///1Ptz4nJweTJ09GfHw8IiMjMWDAABw7dkxxjKNHj0ZUVBSSk5OxY8cORf1Lly5Fnz59jPar5dtvv4WXlxc6dOigWJ6QkKDqjfbtt98iISFBsezgwYPo378/WrRogdatW2P69Om6a7po0SJ07twZBQUFAIB79+6hTZs2Zr06/vjjD4SGhgIAMjIybDIY3bt3D6+88grat2+P8PBwDBw4UNcPx48fj40bN2Lv3r0ICwvD1atXbdoekLywVq5ciTFjxiAiIgJt27bFjz/+qNh+5syZaN++PSIiIjBw4EAcP35csf2XX36JCRMmICoqyuK5yMrKwpQpU9CyZUvExsbiX//6F+7duwdA6jtz5sxBu3btEBkZiR49emDPnj26bS9evIgRI0YgJiYGUVFRGDJkCE6fPq1bv3PnTvTr1093ny1YsEB3n+Xm5mLmzJmIj49HREQEunXrZtLIrL0HtmzZgt69e6N58+bo3r07zpw5oytz/vx5pKWl6e6h8ePH67zHrLmHgJKPGw8ePEBkZCQ2b96sKD9//nydx1yXLl0giqLiWhIEQRCEVXCCIAiCqMAkJCTwVatWGS0fN24cnzRpEuec8127dvHIyEj+66+/8sLCQv7777/zmJgYvnXrVs4555s2beItWrTgf//9NxdFkW/evJk3btyY//3335xzzj/44APetWtXzjnnp06d4tHR0Xz79u2q7bl48SIPCQnhGzZs4Pn5+Xz37t28RYsWuu2vXr3KmzRpwjdv3sw55/zcuXM8Ojqar127lnPO+YEDB3hwcDC/c+cO55zz4OBgvmXLFn7lyhUeEhLCjx07ptvX2rVreUJCAhdFUdGGq1ev8uDgYH779m3OOefffvst79KlCx82bJiuzJgxY/jChQt5Xl4e79ChA3/99df5vXv3+O3bt/mwYcP4kCFDOOecX758mQcHB/O0tDSekZHBRVFUtPHu3bu8W7du/K233tLVrW0z55wvWLCAp6Sk8HPnzvG8vDz+2Wef8dDQUH7p0iXOOeevvPIKHzBgAM/JyeF5eXl8zpw5vF27drq6pkyZwvv06cNv377NMzMz+UsvvaSo3xJpaWl8+vTpimUJCQl8586dPCIigp8+fVq3/Pfff+cJCQl8/fr1fPDgwZxzzh8+fMijoqL46tWreVFREb9x4wZPTk7mCxcu5JxznpeXx7t168aXL1/OOed8zpw5fMiQIUbXhHOpn6WmpvK4uDjevXt3npqayjt16sTbtWvHU1NT+cOHDy0ez4QJE/iAAQP4jRs3+MOHD/nbb7/NY2JieE5ODuec83/961985MiRJd4+ISGBx8fH819++YXn5eXxNWvW8MaNG/PLly9zzqX7avjw4fzWrVs8NzeXf/jhhzwuLk7X9oSEBJ6QkMD379/PCwoK+IoVK3hoaCjPyMhQbc+YMWP4iBEjeGZmJr9z5w7v168ff/XVVznnnC9btownJSXxmzdv8sLCQr569WoeERGha+vw4cP5tGnTeG5uLs/NzeULFizg/fr145xz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HAkJCejUqZPe9mlpadizZw+OHDkCAFCr1XBwcCid4ImIiIiKgEUjIiIiIgNUrlwZaWlp2sfJyckAgLVr1+K1117D5MmTIZfL4evrm+v2Tk5OcHd3Z3c0IiIiMhvsnkZERERkgMaNGyM8PBzJycmQJAmBgYEAgKdPn6J+/fqQy+UICQnBo0ePkJ6eDgCwsLDQFpo8PDwQGBgIpVIJAPjnn39w/Phxk5wLERERkSFkQghh6iCIiIiIzMHff/+NEydOoHLlymjfvj3++ecfDBo0COvXr4e9vT08PDzg6OiIzZs3Y/Lkybhz5w42bNgAb29vfPrpp9i+fTtOnjwJAKhevTq++OILODo6mvisiIiIiHLHohEREREREREREelh9zQiIiIiIiIiItLDohEREREREREREelh0YiIiIiIiIiIiPSwaERERERERERERHpYNCIiIiIiIiIiIj0sGhERERERERERkR4WjYiIiIiIiIiISA+LRkREREREREREpIdFIyIiogpk586daNOmjanDMCvLli1Dz549S2z/6enp6NWrF7Zs2VKk7f38/DB69OgC19u/fz88PT3Ro0ePIh2nIPfv30eTJk1w48aNEtk/AGzfvh1ubm4ltn9DrVixAn369DFo3SZNmuDgwYMlHBEREVHJYNGIiIgqnOjoaDRt2hT9+vUzdShGtXPnTri4uGj/a9KkCZo1a6Z9PHToUPTu3Rtnz54ttZiOHDmCmzdvFnl7Ly8vnXNwcXHB22+/jQkTJuDBgwdGjNR05s+fj3r16qF///4A9F+zs2fP4uLFi3luP2nSJNy+fRsbN27M9zirV69G586dsXfvXuMEDk0RJzY21mj7K6uSk5N1Xl8fHx9s3769xI97//597Nmzp8SPQ0RElBcWjYiIqMLZtGkTunTpghs3biA8PNzU4RhN7969ERoaqv3P3t4ec+fO1T7+9ddfSz2mn3/+uVhFIwD4+uuvdc5r8+bNSElJwahRo6BWq40UqWncuXMHW7duxZdffqldlvM1W7duHS5dupTnPqytrfHFF19g+fLlePbsWZ7rJScno379+pDJZEaJXa1WY/78+YiLizPK/sqyM2fO4K+//ir14x4+fNioRT4iIqLCYtGIiIgqlIyMDOzYsQP9+/fHO++8g02bNmmf++uvv+Dp6QlJkrTL0tPT4ebmhv379wMAjh07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IiIiIiIiIiIiPSwaERERERERERGRHhaNiIiIiIiIiIhID4tGRERERERERESkh0UjIiIiIiIiIiLSw6IRERERERERERHpYdGIiIiIiIiIiIj0sGhERERERERERER6WDQiIiIiIiIiIiI9LBoREREREREREZEeFo2IiIiIiIiIiEgPi0ZERERERERERKSHRSMiIiIiIiIiItLDohEREREREREREemxNGSljIwMTJgwAX379sWbb74Jf39/SJIER0dHjB07FlZWVjh58iT2798PmUwGb29veHl5QaVSYcWKFYiPj4dcLoePjw+qV6+O6OhorFmzBjKZDHXr1sWIESNK+jyJiIiIiIiIiKgQDGpptG3bNlSuXBkAsHnzZnTr1g1z5sxBjRo1EBAQgPT0dGzduhW+vr6YNWsW9u3bh5SUFJw6dQr29vaYO3cu+vTpgz///BMAsH79egwZMgRz585FWloaLl26VHJnSEREREREREREhVZg0ejBgwe4f/8+3NzcAABXr16Fu7s7AMDd3R0hISG4efMmGjRoAHt7e1hbW6NJkyYIDw9HWFgYWrduDQBwcXFBREQEVCoV4uLi0LBhQwBAq1atEBoaWlLnR0RERERERERERVBg97Tff/8dw4YNw/HjxwEASqUSVlZWAAAHBwckJSUhKSkJDg4O2m1yWy6XyyGTyZCUlIRKlSpp133ppZfw5MkTg4J1dnY2yjpEpsY8JXPAPDWOx7Z2eAbA0soKNfmaGh3zlMwB85TMAfOUzAHztPTlWzQ6ceIEGjdujFdffdUoBxNCGLQsLw8fPsz3eWdn5wLXITI15imZA+ap8ajTnwEAVJmZfE2NjHlK5oB5SuaAeUrmgHlacvIrxuVbNLp48SLi4uJw8eJFJCQkwMrKCra2tsjIyIC1tTUSExPh5OQEJycnJCUlabdLTExEo0aNdJarVCoIIeDo6AiFQqGzrpOTU3HPkYiIiIiIiIiIjCjfotH48eO1/968eTNeffVVREREICgoCB06dEBQUBBcXV3RqFEjrFq1CqmpqbCwsEBERASGDBmCZ8+eade5cOECmjVrBktLS9SqVQvh4eFo2rQpgoOD8e6775b4iRIRERERERERkeEKHNMopwEDBsDf3x9HjhxBtWrV0LFjR1haWuKTTz6Bn58fZDIZ+vXrB3t7e3h6eiIkJAS+vr6wsrKCj48PAGDIkCFYvXo1hBBo2LAhmjdvbvQTIyIiIiIiIiKiojO4aDRgwADtv319ffWeb9u2Ldq2bauzTC6XawtF2dWuXRtz5swpTJxERERERERERFSK5KYOgIiIiIiIiIiIyh4WjYiIiIiIiIiISA+LRkREREREREREpIdFIyIiIiIiIiIi0sOiERERERERERER6WHRiIiIiIiIiIiI9LBoREREREREREREelg0IiIiIiIiIiIiPSwaERERERERERGRHhaNiIiIiIiIiIhID4tGRERERERERESkh0UjIiIiIiIiIiLSw6IRUTklUhUQIecghDB1KEQVmgwyU4dARERERFQkLBoRlVPSklmQls0Frl02dShEFZoAC7dEREREZJ5YNCIqr6IjAQAi7pGJAyEiIiIiIiJzxKIRERERERERERHpYdGIiIiIiIiIiIj0sGhERERERERERER6WDQiIiIiIiIiIiI9LBoREREREREREZEeFo2Iyj1O901EpUfcjYLIUJo6DCIiIiIyAhaNiIiIqMjE/dsQcQ81/44KhzR3PKTl35k4KiIiIiIyBhaNiIioQhPxMZAObIOQ1KYOxSxJs8dBmv45AEDcj9YsvHbJdAERERERkdFYmjoAIiIiU5K+nwwkJwHVqkPm8ZapwyEiIiIiKjMMKhpt2LAB169fhyRJ6N27Nxo0aAB/f39IkgRHR0eMHTsWVlZWOHnyJPbv3w+ZTAZvb294eXlBpVJhxYoViI+Ph1wuh4+PD6pXr47o6GisWbMGMpkMdevWxYgRI0r6XIkqJsExjYjylZyk+ZnytER2L4OMI4sRERERkVkqsHtaWFgY7t27Bz8/P3zzzTf47bffsHnzZnTr1g1z5sxBjRo1EBAQgPT0dGzduhW+vr6YNWsW9u3bh5SUFJw6dQr29vaYO3cu+vTpgz///BMAsH79egwZMgRz585FWloaLl1iU/aiEPdvQyhK5kKHiKhCkYpW2hHxMRDXr+T9PEtGRERERGSmCiwavfHGGxg/fjwAoFKlSlAqlbh69Src3d0BAO7u7ggJCcHNmzfRoEED2Nvbw9raGk2aNEF4eDjCwsLQunVrAICLiwsiIiKgUqkQFxeHhg0bAgBatWqF0NDQkjrHcktkKDVjSUwaYupQiIjMn5CKtJn0zUhIi3wh0p8ZOSAiIiIiItMqsGgkl8tha2sL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hi/F/j3gJ7h1M4SCKK8Pq7oSR4c0+1lWXGTXP1NS0cFA3CqbyuWm3Rnn4GjlIg/HujYbaTR79mzS0tI45xw7seIvfvELpk+fTlJSUqP7nAgjjYSQciqOB1JOxfFAyqk4Hkg5FccDKafieCDl9MhpavBN69ZPbYURI0awdKm9esW2bdtIS0trMmAkhBBCCCGEEEIIIY6eNpueNnDgQLKysrjnnntQSnH99de31akIIYQQQgghhBBCiAbaNKfRj3/847b8eCGEEEIIIYQQQgjRiDbLaSSEEEIIIYQQQgghjl1tltNICCGEEEIIIYQQQhy7JGgkhBBCCCGEEEIIIeJI0EgIIYQQQgghhBBCxJGgkRBCCCGEEEIIIYSII0EjIYQQQgghhBBCCBFHgkZCCCGEEEIIIYQQIo6zrU+gOV566SU2bdqEZVlccsklZGdn8/TTT2NZFh06dOCXv/wlLpeLRYsW8f7776OU4uyzz2by5MmUlJQwY8YMwuEwlmVx9dVXk5WV1daXJE5AzS2nVVVVPPnkk3i9Xn7zm98AEA6HefbZZyksLMQwDG6++Wa6du3axlckTkStKaemaTJjxgz279+PZVlcddVVDBo0qI2vSJyIWlNODygrK+O2227j9ttv56STTmqjKxEnstaW07fffptFixbhdDq5/vrr6devXxtejThRtaacSjtKHC3NLaeLFy/mnXfewTAMhg4dypVXXintqKPA8cADDzzQ1ifRlPXr17N8+XLuv/9+xo0bx/Tp0ykqKuK0005jypQpbN++nYKCAnr06MFTTz3Fgw8+yOTJk5kxYwYTJkzgrbfeYtCgQVx33XVkZGQwe/ZsTj/99La+LHGCaW45zc7O5plnniE7O5uysjLGjx8PwMKFC6msrOT222+nY8eOvP3225x66qltfFXiRNPacrpgwYJIOR0wYADPPfccZ599dhtflTjRtLacHvDcc89hmiYjRoygS5cubXQ14kTV2nKal5fH66+/ziOPPEJ2djYrV66U4KY47FpbTufMmSPtKHHENbec9uzZk0cffZRp06Zx7rnn8sorrzBw4EDWrFkj7agj7JifnjZkyBBuu+02AJKTkwkEAmzYsIExY8YAMGbMGNatW0dubi7Z2dn4fD7cbjcDBw4kJyeH1NRUKisrAaiurqZdu3Ztdi3ixNXccgrw85//PG50xvr1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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Monthly-Case-Statistics">Monthly Case Statistics<a class="anchor-link" href="#Monthly-Case-Statistics">&#182;</a></h2><ul>
<li>use groupby() and agg()</li>
<li>mean ttr possible artifact created by dropping open cases the end of the dataset</li>
<li><em>NOTE: ttr is 'looking ahead' contains information about future state</em></li>
</ul>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">

</div>
<div class="cell border-box-sizing code_cell rendered">

<div class="output_wrapper">
<div class="output">

<div class="output_area">



<div class="output_png output_subarea ">
<img src="data:image/png;base64,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bCwAYMqUKUhNTcX+/fsRHh6O6OhofPrppwgICGhz7iVLliAhIQGenp5YtmwZkpOTAdwJDq+99hpUKhX+85//ID4+Hl5eXoiNjcXly5exceNGzJo1C9u2bcPBgwehUCgQFBSEX//613BxccHvfvc7JCQkQETg4eGBefPmWeCVAFQqFf70pz9h165d0Ov18PDwwB//+EfT1wrvJywsDJs3b8bt27cf+i8kiIiILIF5gnmCiIjIXMwTzBNE1qQQEbH2JIgc1c6dOzFgwAC88MILFj3v9u3bERMT0+raeERERPRkYp4gIiIiczFPEDkuXmKIyEp++uknfPvttxg7dqzFzx0aGsoPXyIiIgfAPEFERETmYp4gcmz8BgFRF0lKSsKNGzfa/T9/f3+UlpZizpw5GDly5GOemWXt2rUL586da/f/5s6da/qaIxERET085gnmCSIiInMxTzBPEHWEGwRERERERERERERERA6IlxgiIiIiIiIiIiIiInJA3CAgIiIiIiIiIiIiInJA3CAgIiIiIiIiIiIiInJA3CAgIiIiIiIiIiIiInJA3CAgIiIiIiIiIiIiInJA3CAgIiIiIiIiIiIiInJA/wMTulymHkFYNwAAAABJRU5ErkJggg==
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="How-can-we-model-the-ttr-and-predict-the-resolution-time-of-a-case-opened?">How can we model the ttr and predict the resolution time of a case opened?<a class="anchor-link" href="#How-can-we-model-the-ttr-and-predict-the-resolution-time-of-a-case-opened?">&#182;</a></h3><p>Since the variance of the target is so high, and thinking about the wide range of request types being serviced, it seems reasonable to divide response time into a few classes. Doing so we may discover that certain types of requests tend to be resolved faster in general, or that there are other presently unknown factors that influence how soon our complaints are attended to..</p>
<h3 id="We-can-explore-the-boundaries-for-the-classes.">We can explore the boundaries for the classes.<a class="anchor-link" href="#We-can-explore-the-boundaries-for-the-classes.">&#182;</a></h3><p>Everyone likes a <em>low, medium, high</em> system. We also preview the effect of dropping extreme values.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">

<div class="output_wrapper">
<div class="output">

<div class="output_area">




 
 
<div id="4ad966d1-18eb-4664-8e29-ee224cabbdc0"></div>
<div class="output_subarea output_widget_view ">
<script type="text/javascript">
var element = $('#4ad966d1-18eb-4664-8e29-ee224cabbdc0');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"model_id": "b780f1dd49dc4985aeecbc8f1c1461c6", "version_major": 2, "version_minor": 0}
</script>
</div>

</div>

<div class="output_area">




 
 
<div id="e5bf89b4-fb3c-4602-ab7d-60a788f3a8e2"></div>
<div class="output_subarea output_widget_view ">
<script type="text/javascript">
var element = $('#e5bf89b4-fb3c-4602-ab7d-60a788f3a8e2');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"model_id": "e9a0e2f5e922495baf78b8e238f275cd", "version_major": 2, "version_minor": 0}
</script>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Assign-the-classes-according-to-(my)-common-sense-heuristic:">Assign the classes according to (my) common sense heuristic:<a class="anchor-link" href="#Assign-the-classes-according-to-(my)-common-sense-heuristic:">&#182;</a></h3><p>'low' ttr = 4 hr or less
'med' ttr = between 4 and 40 hours
'high'ttr = anything above 40 hrs</p>
<p>I loathe to drop rows if they dont represent bad data, and there is justification for some of the long ttrs.</p>

<pre><code>df= wrangler.working.copy()
df.ttr = df.ttr.dt.total_seconds()/3600
df[df.ttr &gt; xb].Category.value_counts().head(6)

</code></pre>
<ul>
<li>Graffiti                        1873</li>
<li>Street and Sidewalk Cleaning     610</li>
<li>Other                            370</li>
<li>Damaged Property                 348</li>
<li>Encampments                      232</li>
<li>Street Defects                   163 </li>
</ul>
<p>So we will add the target classes to as above with out dropping, yielding the following balance:</p>

<pre><code>high    0.450375
med     0.285966
low     0.263659</code></pre>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">

<div class="output_wrapper">
<div class="output">

<div class="output_area">



<div class="output_text output_subarea output_execute_result">
<pre>3        high
4         med
5        high
7         med
8        high
         ... 
41677     med
41679     low
41680     low
41681     med
41682     low
Name: ttr_class, Length: 30035, dtype: object</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Time-to-predict-the-future....">Time to predict the future....<a class="anchor-link" href="#Time-to-predict-the-future....">&#182;</a></h3><p>We have simplifed our problem to predicting whether the case we open today will be resolved in one of three time intervals, making this a classfication problem. Our first approach will be to fit a logistic regression, followed by some tree based classifiers. At this point we'll drop <em>CaseID, Status, Notes, and any TimeStamped columns</em> that can either uniquely identify the case, or contain information that was not known at the time of creation. We can retain some features like hour of the day, day of week.</p>

<pre><code>features = df.columns.drop(['CaseID','Opened','Closed','Updated','Status',
'Status Notes','ttr','workload','ttr_class']) #drop time or status related columns</code></pre>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">

<div class="output_wrapper">
<div class="output">

<div class="output_area">



<div class="output_text output_subarea output_execute_result">
<pre>(Index([&#39;Responsible Agency&#39;, &#39;Category&#39;, &#39;Request Type&#39;, &#39;Request Details&#39;,
        &#39;Address&#39;, &#39;Street&#39;, &#39;Neighborhood&#39;, &#39;Police District&#39;, &#39;Latitude&#39;,
        &#39;Longitude&#39;, &#39;Source&#39;, &#39;Media URL&#39;, &#39;Analysis Neighborhoods&#39;,
        &#39;Neighborhoods&#39;, &#39;case_year&#39;, &#39;case_month&#39;],
       dtype=&#39;object&#39;), &#39;ttr_class&#39;)</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">

</div>
<div class="cell border-box-sizing code_cell rendered">

<div class="output_wrapper">
<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">
<pre>(19222, 26) (4806, 26)
</pre>
</div>
</div>

<div class="output_area">



<div class="output_text output_subarea output_execute_result">
<pre>(0.4929255097794424, 0.4899284168470118)</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="...-and-survey-says,-LogReg-not-so-hot.">... and survey says, LogReg not so hot.<a class="anchor-link" href="#...-and-survey-says,-LogReg-not-so-hot.">&#182;</a></h4><p>With validation and test accuracy of (0.4929255097794424, 0.490427834193441), this certainly sets a baseline for further model evaluation.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">

<div class="output_wrapper">
<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">
<pre>decisiontreeclassifier
Training Accuracy: 0.9958901258974092
Validation Accuracy: 0.5624219725343321
Test Accuracy: 0.5568503412685201
baseline Accuracy: 0.45037456300982187 

              precision    recall  f1-score   support

        high       0.68      0.69      0.69      2633
         low       0.55      0.54      0.54      1590
         med       0.38      0.38      0.38      1784

    accuracy                           0.56      6007
   macro avg       0.54      0.54      0.54      6007
weighted avg       0.56      0.56      0.56      6007

randomforestclassifier
Training Accuracy: 0.7205805847466444
Validation Accuracy: 0.643986683312526
Test Accuracy: 0.6435824870983852
baseline Accuracy: 0.45037456300982187 

              precision    recall  f1-score   support

        high       0.67      0.91      0.77      2633
         low       0.65      0.68      0.67      1590
         med       0.51      0.22      0.30      1784

    accuracy                           0.64      6007
   macro avg       0.61      0.60      0.58      6007
weighted avg       0.62      0.64      0.60      6007

xgbclassifier
Training Accuracy: 0.6871293309749246
Validation Accuracy: 0.6437786100707449
Test Accuracy: 0.6445813217912435
baseline Accuracy: 0.45037456300982187 

              precision    recall  f1-score   support

        high       0.67      0.91      0.77      2633
         low       0.64      0.69      0.66      1590
         med       0.52      0.22      0.30      1784

    accuracy                           0.64      6007
   macro avg       0.61      0.60      0.58      6007
weighted avg       0.62      0.64      0.60      6007

</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">

<div class="output_wrapper">
<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">
<pre>xgbclassifier
[[1921   58  139]
 [ 258  888  155]
 [ 680  422  285]]
</pre>
</div>
</div>

<div class="output_area">



<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">

<div class="output_wrapper">
<div class="output">

<div class="output_area">



<div class="output_text output_subarea output_execute_result">
<pre>[&#39;Responsible Agency&#39;,
 &#39;Category&#39;,
 &#39;Request Type&#39;,
 &#39;Request Details&#39;,
 &#39;Address&#39;,
 &#39;Street&#39;,
 &#39;Neighborhood&#39;,
 &#39;Police District&#39;,
 &#39;Latitude&#39;,
 &#39;Longitude&#39;,
 &#39;Source&#39;,
 &#39;Media URL&#39;,
 &#39;Analysis Neighborhoods&#39;,
 &#39;Neighborhoods&#39;,
 &#39;case_year&#39;,
 &#39;case_month&#39;]</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Linear-regression-lite,-Plotly-Express-with-OLS-trendline-:">Linear regression lite, Plotly Express with OLS trendline :<a class="anchor-link" href="#Linear-regression-lite,-Plotly-Express-with-OLS-trendline-:">&#182;</a></h3><p>Without doing much more than a simple scatter plot of workload, we can visualize a regression line that captures the general upward trend inside of the undulation.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">

<div class="output_wrapper">
<div class="output">

<div class="output_area">


<div class="output_html rendered_html output_subarea ">
<div>
        
        
            <div id="f60bc0f6-d1d8-4444-9d52-b798310f4069" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    
                if (document.getElementById("f60bc0f6-d1d8-4444-9d52-b798310f4069")) {
                    Plotly.newPlot(
                        'f60bc0f6-d1d8-4444-9d52-b798310f4069',
                        [{"hovertemplate": "day_offset=%{x}<br>workload=%{y}<extra></extra>", "legendgroup": "", "marker": {"color": "#636efa", "symbol": "circle"}, "mode": "markers", "name": "", "showlegend": false, "type": "scattergl", "x": [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 101, 102, 103, 104, 105, 106, 107, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 179, 180, 181, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 309, 311, 313, 314, 315, 316, 318, 319, 321, 322, 323, 324, 325, 326, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 339, 340, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 383, 384, 385, 386, 387, 388, 390, 391, 392, 393, 395, 396, 397, 398, 399, 400, 401, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 430, 431, 433, 434, 435, 436, 438, 439, 440, 442, 443, 444, 445, 446, 447, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 461, 462, 463, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 661, 662, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 860, 861, 862, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 892, 893, 894, 895, 896, 897, 898, 899, 901, 902, 903, 904, 905, 907, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 972, 973, 974, 975, 976, 978, 979, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1082, 1084, 1085, 1086, 1087, 1088, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1112, 1113, 1114, 1115, 1116, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240, 1241, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249, 1250, 1252, 1253, 1254, 1255, 1257, 1258, 1259, 1260, 1261, 1262, 1263, 1264, 1265, 1266, 1267, 1268, 1269, 1270, 1271, 1272, 1273, 1274, 1275, 1276, 1277, 1279, 1280, 1281, 1282, 1283, 1284, 1285, 1286, 1287, 1288, 1289, 1290, 1291, 1292, 1293, 1294, 1295, 1296, 1297, 1298, 1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309, 1310, 1311, 1312, 1315, 1316, 1317, 1318, 1319, 1320, 1321, 1322, 1323, 1324, 1325, 1326, 1328, 1329, 1330, 1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341, 1342, 1343, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1359, 1360, 1361, 1362, 1363, 1364, 1365, 1366, 1367, 1368, 1369, 1370, 1371, 1372, 1373, 1374, 1375, 1376, 1378, 1379, 1380, 1381, 1382, 1383, 1384, 1385, 1386, 1387, 1388, 1389, 1391, 1392, 1393, 1394, 1395, 1396, 1397, 1398, 1399, 1400, 1401, 1402, 1403, 1404, 1406, 1407, 1408, 1409, 1410, 1412, 1413, 1414, 1415, 1416, 1417, 1418, 1419, 1420, 1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428, 1429, 1430, 1431, 1432, 1433, 1434, 1435, 1436, 1437, 1438, 1439, 1440, 1441, 1442, 1443, 1444, 1445, 1446, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456, 1457, 1458, 1459, 1460, 1461, 1462, 1463, 1464, 1465, 1466, 1467, 1468, 1469, 1470, 1471, 1472, 1473, 1474, 1475, 1476, 1477, 1478, 1479, 1480, 1482, 1483, 1484, 1485, 1486, 1487, 1488, 1490, 1491, 1492, 1493, 1494, 1495, 1496, 1497, 1498, 1499, 1500, 1501, 1502, 1503, 1504, 1505, 1506, 1507, 1508, 1509, 1510, 1511, 1512, 1513, 1514, 1515, 1516, 1517, 1518, 1519, 1520, 1521, 1522, 1523, 1524, 1525, 1526, 1527, 1528, 1529, 1530, 1531, 1532, 1533, 1534, 1535, 1536, 1537, 1538, 1539, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549, 1550, 1551, 1552, 1553, 1554, 1555, 1556, 1557, 1558, 1560, 1561, 1562, 1563, 1564, 1565, 1566, 1567, 1568, 1569, 1570, 1571, 1572, 1573, 1574, 1575, 1576, 1577, 1578, 1579, 1580, 1581, 1582, 1583, 1584, 1585, 1586, 1587, 1588, 1589, 1591, 1592, 1593, 1594, 1595, 1596, 1597, 1600, 1601, 1602, 1603, 1605, 1606, 1607, 1608, 1609, 1610, 1611, 1612, 1613, 1614, 1615, 1616, 1617, 1618, 1619, 1620, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 1635, 1636, 1637, 1638, 1639, 1640, 1641, 1642, 1643, 1644, 1645, 1646, 1647, 1648, 1649, 1650, 1651, 1652, 1653, 1654, 1655, 1656, 1657, 1658, 1659, 1660, 1661, 1662, 1663, 1664, 1665, 1666, 1667, 1668, 1669, 1670, 1671, 1672, 1673, 1674, 1675, 1676, 1677, 1679, 1680, 1681, 1682, 1683, 1684, 1685, 1686, 1687, 1688, 1689, 1690, 1691, 1692, 1693, 1694, 1695, 1696, 1697, 1698, 1699, 1700, 1701, 1702, 1703, 1704, 1705, 1706, 1707, 1708, 1709, 1710, 1711, 1712, 1713, 1714, 1715, 1716, 1717, 1718, 1719, 1720, 1721, 1722, 1723, 1724, 1725, 1726, 1727, 1728, 1729, 1730, 1731, 1732, 1733, 1735, 1736, 1737, 1738, 1739, 1740, 1741, 1742, 1743, 1744, 1745, 1746, 1747, 1748, 1749, 1750, 1751, 1752, 1753, 1754, 1755, 1756, 1757, 1758, 1759, 1760, 1761, 1762, 1763, 1764, 1765, 1766, 1767, 1768, 1769, 1770, 1771, 1772, 1773, 1774, 1775, 1776, 1777, 1778, 1779, 1780, 1781, 1782, 1783, 1784, 1785, 1786, 1787, 1788, 1789, 1790, 1791, 1792, 1793, 1794, 1795, 1796, 1797, 1798, 1799, 1800, 1801, 1802, 1803, 1804, 1805, 1806, 1807, 1808, 1809, 1810, 1811, 1812, 1813, 1814, 1815, 1816, 1817, 1818, 1819, 1820, 1821, 1822, 1823, 1824, 1825, 1826, 1827, 1828, 1829, 1830, 1831, 1832, 1833, 1834, 1835, 1836, 1837, 1838, 1839, 1840, 1841, 1842, 1843, 1844, 1845, 1847, 1848, 1849, 1850, 1851, 1852, 1853, 1854, 1855, 1856, 1857, 1858, 1859, 1860, 1861, 1862, 1863, 1864, 1865, 1866, 1867, 1868, 1869, 1870, 1871, 1872, 1873, 1874, 1875, 1876, 1877, 1878, 1879, 1880, 1881, 1882, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 1890, 1891, 1892, 1893, 1894, 1895, 1896, 1897, 1898, 1899, 1900, 1901, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910, 1911, 1912, 1913, 1914, 1915, 1916, 1917, 1918, 1919, 1920, 1921, 1922, 1923, 1924, 1925, 1926, 1927, 1928, 1929, 1930, 1931, 1932, 1933, 1934, 1935, 1936, 1937, 1938, 1939, 1940, 1941, 1942, 1943, 1944, 1945, 1946, 1947, 1948, 1949, 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 2058, 2059, 2060, 2061, 2062, 2063, 2064, 2065, 2066, 2067, 2068, 2069, 2070, 2071, 2072, 2073, 2074, 2075, 2076, 2077, 2078, 2079, 2080, 2081, 2082, 2083, 2084, 2085, 2086, 2087, 2088, 2089, 2090, 2091, 2092, 2093, 2094, 2095, 2096, 2097, 2098, 2099, 2100, 2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112, 2113, 2114, 2115, 2116, 2117, 2118, 2119, 2120, 2121, 2122, 2123, 2124, 2125, 2126, 2127, 2128, 2129, 2130, 2131, 2132, 2133, 2134, 2135, 2136, 2137, 2138, 2139, 2140, 2141, 2142, 2143, 2144, 2145, 2146, 2147, 2148, 2149, 2150, 2151, 2152, 2153, 2154, 2155, 2156, 2157, 2158, 2159, 2160, 2161, 2162, 2163, 2164, 2165, 2166, 2167, 2168, 2169, 2170, 2171, 2172, 2173, 2174, 2175, 2176, 2177, 2178, 2179, 2180, 2181, 2182, 2183, 2184, 2185, 2186, 2187, 2188, 2189, 2190, 2191, 2192, 2193, 2194, 2195, 2196, 2197, 2198, 2199, 2200, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212, 2213, 2214, 2215, 2216, 2217, 2218, 2219, 2220, 2221, 2222, 2223, 2224, 2225, 2226, 2227, 2228, 2229, 2230, 2231, 2232, 2233, 2234, 2235, 2236, 2237, 2238, 2239, 2240, 2241, 2242, 2243, 2244, 2245, 2246, 2247, 2248, 2249, 2250, 2251, 2252, 2253, 2254, 2255, 2256, 2257, 2258, 2259, 2260, 2261, 2262, 2263, 2264, 2265, 2266, 2267, 2268, 2269, 2270, 2271, 2272, 2273, 2274, 2275, 2276, 2277, 2278, 2279, 2280, 2281, 2282, 2283, 2284, 2285, 2286, 2287, 2288, 2289, 2290, 2291, 2292, 2293, 2294, 2295, 2296, 2297, 2298, 2299, 2300, 2301, 2302, 2303, 2304, 2305, 2306, 2307, 2308, 2309, 2310, 2311, 2312, 2313, 2314, 2315, 2316, 2317, 2318, 2319, 2320, 2321, 2322, 2323, 2324, 2325, 2326, 2327, 2328, 2329, 2330, 2331, 2332, 2333, 2334, 2335, 2336, 2337, 2338, 2339, 2341, 2342, 2343, 2344, 2345, 2346, 2347, 2348, 2349, 2350, 2351, 2352, 2353, 2354, 2355, 2356, 2357, 2358, 2359, 2360, 2361, 2362, 2363, 2364, 2365, 2366, 2367, 2368, 2369, 2370, 2371, 2372, 2373, 2374, 2375, 2376, 2377, 2378, 2379, 2380, 2381, 2382, 2383, 2384, 2385, 2386, 2387, 2388, 2389, 2390, 2391, 2392, 2393, 2394, 2395, 2396, 2397, 2398, 2399, 2400, 2401, 2402, 2403, 2404, 2405, 2406, 2407, 2408, 2409, 2410, 2411, 2412, 2413, 2414, 2415, 2416, 2417, 2418, 2419, 2420, 2421, 2422, 2423, 2424, 2425, 2426, 2427, 2429, 2430, 2431, 2432, 2433, 2434, 2435, 2436, 2437, 2438, 2439, 2440, 2441, 2442, 2443, 2444, 2445, 2446, 2447, 2448, 2449, 2450, 2451, 2452, 2453, 2454, 2455, 2456, 2457, 2458, 2459, 2460, 2461, 2462, 2463, 2464, 2465, 2466, 2467, 2468, 2469, 2470, 2471, 2472, 2473, 2474, 2475, 2476, 2477, 2478, 2479, 2480, 2481, 2482, 2483, 2484, 2485, 2486, 2487, 2488, 2489, 2490, 2491, 2492, 2493, 2494, 2495, 2496, 2497, 2498, 2499, 2500, 2501, 2502, 2503, 2504, 2505, 2506, 2507, 2508, 2509, 2510, 2511, 2512, 2513, 2514, 2515, 2516, 2517, 2518, 2519, 2520, 2521, 2522, 2523, 2524, 2525, 2526, 2527, 2528, 2529, 2530, 2531, 2532, 2533, 2534, 2535, 2536, 2537, 2539, 2540, 2541, 2542, 2543, 2544, 2545, 2546, 2547, 2548, 2549, 2550, 2551, 2552, 2553, 2554, 2555, 2556, 2557, 2558, 2559, 2560, 2561, 2562, 2563, 2564, 2565, 2566, 2567, 2568, 2569, 2570, 2571, 2572, 2573, 2574, 2575, 2576, 2577, 2578, 2579, 2580, 2581, 2582, 2583, 2584, 2585, 2586, 2587, 2588, 2589, 2590, 2591, 2592, 2593, 2594, 2595, 2596, 2597, 2598, 2599, 2600, 2601, 2602, 2603, 2604, 2605, 2606, 2607, 2608, 2609, 2610, 2611, 2612, 2613, 2614, 2615, 2616, 2617, 2618, 2619, 2620, 2621, 2622, 2623, 2624, 2625, 2626, 2627, 2628, 2629, 2630, 2631, 2632, 2633, 2634, 2635, 2636, 2637, 2638, 2639, 2640, 2641, 2642, 2643, 2644, 2645, 2646, 2647, 2648, 2649, 2650, 2651, 2652, 2653, 2654, 2655, 2656, 2657, 2658, 2659, 2660, 2661, 2662, 2663, 2664, 2665, 2666, 2667, 2668, 2669, 2670, 2671, 2672, 2673, 2674, 2675, 2676, 2677, 2678, 2679, 2680, 2681, 2682, 2683, 2684, 2685, 2686, 2687, 2688, 2689, 2690, 2691, 2692, 2693, 2694, 2695, 2696, 2697, 2698, 2699, 2700, 2701, 2702, 2703, 2704, 2705, 2706, 2707, 2708, 2709, 2710, 2711, 2712, 2713, 2714, 2715, 2716, 2717, 2718, 2719, 2720, 2721, 2722, 2723, 2724, 2725, 2726, 2727, 2728, 2729, 2730, 2731, 2732, 2733, 2734, 2735, 2736, 2737, 2738, 2739, 2740, 2741, 2742, 2743, 2744, 2745, 2746, 2747, 2748, 2749, 2750, 2751, 2752, 2753, 2754, 2755, 2756, 2757, 2758, 2759, 2760, 2761, 2762, 2763, 2764, 2765, 2766, 2767, 2768, 2769, 2770, 2771, 2772, 2773, 2774, 2775, 2776, 2777, 2778, 2779, 2780, 2781, 2782, 2783, 2784, 2785, 2786, 2787, 2788, 2789, 2790, 2791, 2792, 2793, 2794, 2795, 2796, 2797, 2798, 2799, 2800, 2801, 2802, 2803, 2804, 2805, 2806, 2807, 2808, 2809, 2810, 2811, 2812, 2813, 2814, 2815, 2816, 2817, 2818, 2819, 2820, 2821, 2822, 2823, 2824, 2825, 2826, 2827, 2828, 2829, 2830, 2831, 2832, 2833, 2834, 2835, 2836, 2837, 2838, 2839, 2840, 2841, 2842, 2843, 2844, 2845, 2846, 2847, 2848, 2849, 2850, 2851, 2852, 2853, 2854, 2855, 2856, 2857, 2858, 2859, 2860, 2861, 2862, 2863, 2864, 2865, 2866, 2867, 2868, 2869, 2870, 2871, 2872, 2873, 2874, 2875, 2876, 2877, 2878, 2879, 2880, 2881, 2882, 2883, 2884, 2885, 2886, 2887, 2888, 2889, 2890, 2891, 2892, 2893, 2894, 2895, 2896, 2897, 2898, 2899, 2900, 2901, 2902, 2903, 2904, 2905, 2906, 2907, 2908, 2909, 2910, 2911, 2912, 2913, 2914, 2915, 2916, 2917, 2918, 2919, 2920, 2921, 2922, 2923, 2924, 2925, 2926, 2927, 2928, 2929, 2930, 2932, 2933, 2934, 2935, 2936, 2937, 2938, 2939, 2940, 2941, 2942, 2943, 2944, 2945, 2946, 2947, 2948, 2949, 2950, 2951, 2952, 2953, 2954, 2955, 2956, 2957, 2958, 2959, 2960, 2961, 2962, 2963, 2964, 2965, 2966, 2967, 2968, 2969, 2970, 2971, 2972, 2973, 2974, 2975, 2976, 2977, 2978, 2979, 2980, 2981, 2982, 2983, 2984, 2985, 2986, 2987, 2988, 2989, 2990, 2991, 2992, 2993, 2994, 2995, 2996, 2997, 2998, 2999, 3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 3012, 3013, 3014, 3015, 3016, 3017, 3018, 3019, 3020, 3021, 3022, 3023, 3024, 3025, 3026, 3027, 3028, 3029, 3030, 3031, 3032, 3033, 3034, 3035, 3036, 3037, 3038, 3039, 3040, 3041, 3042, 3043, 3044, 3045, 3046, 3047, 3048, 3049, 3050, 3051, 3052, 3053, 3054, 3055, 3056, 3057, 3058, 3059, 3060, 3061, 3062, 3063, 3064, 3065, 3066, 3067, 3068, 3069, 3070, 3071, 3072, 3073, 3074, 3075, 3076, 3077, 3078, 3079, 3080, 3081, 3082, 3083, 3084, 3085, 3086, 3087, 3088, 3089, 3090, 3091, 3092, 3093, 3094, 3095, 3096, 3097, 3098, 3099, 3100, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110, 3111, 3112, 3113, 3114, 3115, 3116, 3117, 3118, 3119, 3120, 3121, 3122, 3123, 3124, 3125, 3126, 3127, 3128, 3129, 3130, 3131, 3132, 3133, 3134, 3135, 3136, 3137, 3138, 3139, 3140, 3141, 3142, 3143, 3144, 3145, 3146, 3147, 3148, 3149, 3150, 3151, 3152, 3153, 3154, 3155, 3156, 3157, 3158, 3159, 3160, 3161, 3162, 3163, 3164, 3165, 3166, 3167, 3168, 3169, 3170, 3171, 3172, 3173, 3174, 3175, 3176, 3177, 3178, 3179, 3180, 3181, 3182, 3183, 3184, 3185, 3186, 3187, 3188, 3189, 3190, 3191, 3192, 3193, 3194, 3195, 3196, 3197, 3198, 3199, 3200, 3201, 3202, 3203, 3204, 3205, 3206, 3207, 3208, 3209, 3210, 3211, 3212, 3213, 3214, 3215, 3216, 3217, 3218, 3219, 3220, 3221, 3222, 3223, 3224, 3225, 3226, 3227, 3228, 3229, 3230, 3231, 3232, 3233, 3234, 3235, 3236, 3237, 3238, 3239, 3240, 3241, 3242, 3243, 3244, 3245, 3246, 3247, 3248, 3249, 3250, 3251, 3252, 3253, 3254, 3255, 3256, 3257, 3258, 3259, 3260, 3261, 3262, 3263, 3264, 3265, 3266, 3267, 3268, 3269, 3270, 3271, 3272, 3273, 3274, 3275, 3276, 3277, 3278, 3279, 3280, 3281, 3282, 3283, 3284, 3285, 3286, 3287, 3288, 3289, 3290, 3291, 3292, 3293, 3294, 3295, 3296, 3297, 3298, 3299, 3300, 3301, 3302, 3303, 3304, 3305, 3306, 3307, 3308, 3309, 3310, 3311, 3312, 3313, 3314, 3315, 3316, 3317, 3318, 3319, 3320, 3321, 3322, 3323, 3324, 3325, 3326, 3327, 3328, 3329, 3330, 3331, 3332, 3333, 3334, 3335, 3336, 3337, 3338, 3339, 3340, 3341, 3342, 3343, 3344, 3345, 3346, 3347, 3348, 3349, 3350, 3351, 3352, 3353, 3354, 3355, 3356, 3357, 3358, 3359, 3360, 3361, 3362, 3363, 3364, 3365, 3366, 3367, 3368, 3369, 3370, 3371, 3372, 3373, 3374, 3375, 3376, 3377, 3378, 3379, 3380, 3381, 3382, 3383, 3384, 3385, 3386, 3387, 3388, 3389, 3390, 3391, 3392, 3393, 3394, 3395, 3396, 3397, 3398, 3399, 3400, 3401, 3402, 3403, 3404, 3405, 3406, 3407, 3408, 3409, 3410, 3411, 3412, 3413, 3414, 3415, 3416, 3417, 3418, 3419, 3420, 3421, 3422, 3423, 3424, 3425, 3426, 3427, 3428, 3429, 3430, 3431, 3432, 3433, 3434, 3435, 3436, 3437, 3438, 3439, 3440, 3441, 3442, 3443, 3444, 3445, 3446, 3447, 3448, 3449, 3450, 3451, 3452, 3453, 3454, 3455, 3456, 3457, 3458, 3459, 3460, 3461, 3462, 3463, 3464, 3465, 3466, 3467, 3468, 3469, 3470, 3471, 3472, 3473, 3474, 3475, 3476, 3477, 3478, 3479, 3480, 3481, 3482, 3483, 3484, 3485, 3486, 3487, 3488, 3489, 3490, 3491, 3492, 3493, 3494, 3495, 3496, 3497, 3498, 3499, 3500, 3501, 3502, 3503, 3504, 3505, 3506, 3507, 3508, 3509, 3510, 3511, 3512, 3513, 3514, 3515, 3516, 3517, 3518, 3519, 3520, 3521, 3522, 3523, 3524, 3525, 3526, 3527, 3528, 3529, 3530, 3531, 3532, 3533, 3534, 3535, 3536, 3537, 3538, 3539, 3540, 3541, 3542, 3543, 3544, 3545, 3546, 3547, 3548, 3549, 3550, 3551, 3552, 3553, 3554, 3555, 3556, 3557, 3558, 3559, 3560, 3561, 3562, 3563, 3564, 3565, 3566, 3567, 3568, 3569, 3570, 3571, 3572, 3573, 3574, 3575, 3576, 3577, 3578, 3579, 3580, 3581, 3582, 3583, 3584, 3585, 3586, 3587, 3588, 3589, 3590, 3591, 3592, 3593, 3594, 3595, 3596, 3597, 3598, 3599, 3600, 3601, 3602, 3603, 3604, 3605, 3606, 3607, 3608, 3609, 3610, 3611, 3612, 3613, 3614, 3615, 3616, 3617, 3618, 3619, 3620, 3621, 3622, 3623, 3624, 3625, 3626, 3627, 3628, 3629, 3630, 3631, 3632, 3633, 3634, 3635, 3636, 3637, 3638, 3639, 3640, 3641, 3642, 3643, 3644, 3645, 3646, 3647, 3648, 3649, 3650, 3651, 3652, 3653, 3654, 3655, 3656, 3657, 3658, 3659, 3660, 3661, 3662, 3663, 3664, 3665, 3666, 3667, 3668, 3669, 3670, 3671, 3672, 3673, 3674, 3675, 3676, 3677, 3678, 3679, 3680, 3681, 3682, 3683, 3684, 3685, 3686, 3687, 3688, 3689, 3690, 3691, 3692, 3693, 3694, 3695, 3696, 3697, 3698, 3699, 3700, 3701, 3702, 3703, 3704, 3705, 3706, 3707, 3708, 3709, 3710, 3711, 3712, 3713, 3714, 3715, 3716, 3717, 3718, 3719, 3720, 3721, 3722, 3723, 3724, 3725, 3726, 3727, 3728, 3729, 3730, 3731, 3732, 3733, 3734, 3735, 3736, 3737, 3738, 3739, 3740, 3741, 3742, 3743, 3744, 3745, 3746, 3747, 3748, 3749, 3750, 3751, 3752, 3753, 3754, 3755, 3756, 3757, 3758, 3759, 3760, 3761, 3762, 3763, 3764, 3765, 3766, 3767, 3768, 3769, 3770, 3771, 3772, 3773, 3774, 3775, 3776, 3777, 3778, 3779, 3780, 3781, 3782, 3783, 3784, 3785, 3786, 3787, 3788, 3789, 3790, 3791, 3792, 3793, 3794, 3795, 3796, 3797, 3798, 3799, 3800, 3801, 3802, 3803, 3804, 3805, 3806, 3807, 3808, 3809, 3810, 3811, 3812, 3813, 3814, 3815, 3816, 3817, 3818, 3819, 3820, 3821, 3822, 3823, 3824, 3825, 3826, 3827, 3828, 3829, 3830, 3831, 3832, 3833, 3834, 3835, 3836, 3837, 3838, 3839, 3840, 3841, 3842, 3843, 3844, 3845, 3846, 3847, 3848, 3849, 3850, 3851, 3852, 3853, 3854, 3855, 3856, 3857, 3858, 3859, 3860, 3861, 3862, 3863, 3864, 3865, 3866, 3867, 3868, 3869, 3870, 3871, 3872, 3873, 3874, 3875, 3876, 3877, 3878, 3879, 3880, 3881, 3882, 3883, 3884, 3885, 3886, 3887, 3888, 3889, 3890, 3891, 3892, 3893, 3894, 3895, 3896, 3897, 3898, 3899, 3900, 3901, 3902, 3903, 3904, 3905, 3906, 3907, 3908, 3909, 3910, 3911, 3912, 3913, 3914, 3915, 3916, 3917, 3918, 3919, 3920, 3921, 3922, 3923, 3924, 3925, 3926, 3927, 3928, 3929, 3930, 3931, 3932, 3933, 3934, 3935, 3936, 3937, 3938, 3939, 3940, 3941, 3942, 3943, 3944, 3945, 3946, 3947, 3948, 3949, 3950, 3951, 3952, 3953, 3954, 3955, 3956, 3957, 3958, 3959, 3960, 3961, 3962, 3963, 3964, 3965, 3966, 3967, 3968, 3969, 3970, 3971, 3972, 3973, 3974, 3975, 3976, 3977, 3978, 3979, 3980, 3981, 3982, 3983, 3984, 3985, 3986, 3987, 3988, 3989, 3990, 3991, 3992, 3993, 3994, 3995, 3996, 3997, 3998, 3999, 4000, 4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008, 4009, 4010, 4011, 4012, 4013, 4014, 4015, 4016, 4017, 4018, 4019, 4020, 4021, 4022, 4023, 4024, 4025, 4026, 4027, 4028, 4029, 4030, 4031, 4032, 4033, 4034, 4035, 4036, 4037, 4038, 4039, 4040, 4041, 4042, 4043, 4044, 4045, 4046, 4047, 4048, 4049, 4050, 4051, 4052, 4053, 4054, 4055, 4056, 4057, 4058, 4059, 4060, 4061, 4062, 4063, 4064, 4065, 4066, 4067, 4068, 4069, 4070, 4071, 4072, 4073, 4074, 4075, 4076, 4077, 4078, 4079, 4080, 4081, 4082, 4083, 4084, 4085, 4086, 4087, 4088, 4089, 4090, 4091, 4092, 4093, 4094, 4095, 4096, 4097, 4098, 4099, 4100, 4101, 4102, 4103, 4104, 4105, 4106, 4107, 4108, 4109, 4110, 4111, 4112, 4113, 4114, 4115, 4116, 4117, 4118, 4119, 4120, 4121, 4122, 4123, 4124, 4125, 4126, 4127, 4128, 4129, 4130, 4131, 4132, 4133, 4134, 4135, 4136, 4137, 4138, 4139, 4140, 4141, 4142, 4143, 4144, 4145, 4146, 4147, 4148, 4149, 4150, 4151, 4152, 4153, 4154, 4155, 4156, 4157, 4158, 4159, 4160, 4161, 4162, 4163, 4164, 4165, 4166, 4167, 4168, 4169, 4170, 4171, 4172, 4173, 4174, 4175, 4176, 4177, 4178, 4179, 4180, 4181, 4182, 4183, 4184, 4185, 4186, 4187, 4188, 4189, 4190, 4191, 4192, 4193, 4194, 4195, 4196, 4197, 4198, 4199, 4200, 4201, 4202, 4203, 4204, 4205, 4206, 4207, 4208, 4209, 4210, 4211, 4212, 4213, 4214, 4215, 4216, 4217, 4218, 4219, 4220, 4221, 4222, 4223, 4224, 4225, 4226, 4227, 4228, 4229, 4230, 4231, 4232, 4233, 4234, 4235, 4236, 4237, 4238, 4239, 4240, 4241, 4242, 4243, 4244, 4245, 4246, 4247, 4248, 4249, 4250, 4251, 4252, 4253, 4254, 4255, 4256, 4257, 4258, 4259, 4260, 4261, 4262, 4263, 4264, 4265, 4266, 4267, 4268, 4269, 4270, 4271, 4272, 4273, 4274, 4275, 4276, 4277, 4278, 4279, 4280, 4281, 4282, 4283, 4284, 4285, 4286, 4287, 4288, 4289, 4290, 4291, 4292, 4293, 4294, 4295, 4296, 4297, 4298, 4299, 4300, 4301, 4302, 4303, 4304, 4305, 4306, 4307, 4308, 4309, 4310, 4311, 4312, 4313, 4314, 4315, 4316, 4317, 4318, 4319, 4320, 4321, 4322, 4323, 4324, 4325, 4326, 4327, 4328, 4329, 4330, 4331, 4332, 4333, 4334, 4335, 4336, 4337, 4338, 4339, 4340, 4341, 4342, 4343, 4344, 4345, 4346, 4347, 4348, 4349, 4350, 4351, 4352, 4353, 4354, 4355, 4356, 4357, 4358], "xaxis": "x", "y": [5, 4, 6, 7, 9, 9, 12, 14, 11, 14, 15, 15, 17, 20, 24, 22, 25, 25, 25, 25, 23, 18, 19, 21, 20, 20, 27, 25, 22, 23, 23, 18, 20, 20, 22, 24, 25, 25, 26, 25, 23, 24, 26, 26, 26, 24, 24, 25, 28, 27, 29, 27, 29, 31, 28, 27, 21, 23, 24, 27, 27, 27, 28, 29, 30, 19, 20, 22, 20, 16, 17, 21, 21, 17, 19, 20, 24, 28, 29, 31, 34, 36, 39, 34, 37, 30, 28, 26, 26, 27, 26, 28, 23, 23, 24, 23, 26, 28, 29, 25, 23, 26, 30, 31, 29, 27, 24, 23, 22, 22, 24, 28, 28, 28, 29, 27, 27, 27, 30, 31, 32, 33, 35, 35, 33, 31, 30, 28, 29, 25, 30, 28, 27, 25, 25, 27, 24, 25, 23, 25, 24, 22, 26, 27, 28, 28, 28, 27, 30, 34, 32, 32, 33, 33, 31, 34, 34, 31, 32, 33, 34, 33, 35, 33, 33, 29, 29, 30, 34, 32, 33, 37, 37, 39, 42, 45, 41, 37, 40, 42, 46, 46, 45, 44, 45, 46, 42, 45, 47, 48, 46, 45, 50, 51, 50, 54, 55, 57, 56, 56, 58, 57, 58, 58, 53, 54, 55, 59, 58, 57, 54, 55, 56, 55, 56, 57, 57, 57, 54, 52, 53, 54, 55, 54, 55, 53, 55, 55, 58, 58, 56, 49, 55, 55, 56, 57, 58, 53, 55, 54, 54, 55, 56, 53, 56, 57, 56, 56, 56, 57, 51, 49, 47, 52, 57, 56, 56, 54, 61, 60, 59, 59, 56, 53, 56, 54, 52, 56, 59, 55, 56, 58, 57, 58, 57, 59, 60, 61, 64, 64, 69, 70, 69, 65, 62, 63, 65, 67, 69, 70, 68, 65, 65, 71, 69, 68, 68, 69, 77, 77, 76, 80, 82, 84, 85, 90, 92, 94, 97, 70, 73, 74, 73, 70, 65, 70, 70, 70, 68, 68, 67, 71, 73, 73, 73, 72, 70, 72, 76, 79, 79, 82, 82, 84, 78, 81, 86, 82, 82, 83, 82, 83, 87, 86, 83, 83, 85, 85, 87, 89, 88, 89, 90, 87, 86, 90, 90, 85, 85, 87, 91, 94, 94, 93, 90, 92, 93, 93, 93, 90, 89, 87, 86, 90, 89, 85, 86, 85, 84, 84, 89, 85, 83, 81, 83, 80, 81, 80, 82, 72, 77, 76, 79, 80, 83, 81, 80, 80, 81, 80, 80, 78, 81, 80, 81, 84, 80, 78, 79, 81, 82, 86, 83, 82, 86, 88, 91, 99, 95, 90, 89, 92, 95, 91, 90, 84, 67, 74, 73, 69, 68, 70, 72, 71, 75, 72, 71, 72, 75, 75, 77, 80, 83, 82, 81, 82, 80, 83, 83, 79, 77, 77, 75, 75, 78, 77, 70, 71, 68, 72, 74, 77, 76, 77, 75, 76, 74, 72, 72, 72, 71, 73, 77, 78, 80, 84, 85, 85, 85, 88, 87, 87, 87, 83, 74, 76, 78, 78, 80, 77, 76, 73, 71, 73, 70, 71, 68, 69, 72, 72, 74, 77, 76, 71, 73, 71, 74, 73, 78, 76, 77, 75, 77, 74, 75, 80, 81, 79, 82, 81, 81, 81, 82, 82, 83, 83, 83, 83, 85, 86, 92, 96, 96, 97, 95, 96, 102, 101, 93, 89, 88, 88, 88, 85, 86, 89, 89, 89, 91, 92, 93, 90, 87, 88, 89, 91, 92, 93, 90, 93, 93, 91, 93, 97, 97, 99, 97, 96, 97, 94, 96, 91, 86, 83, 80, 80, 81, 86, 87, 90, 88, 90, 91, 96, 94, 81, 89, 84, 82, 82, 83, 82, 83, 81, 81, 77, 79, 83, 83, 76, 79, 82, 81, 84, 85, 82, 77, 78, 78, 80, 85, 84, 84, 81, 80, 87, 88, 85, 85, 85, 86, 87, 91, 92, 94, 98, 102, 104, 106, 108, 113, 113, 76, 77, 77, 77, 74, 75, 77, 76, 75, 75, 80, 81, 73, 76, 77, 78, 79, 79, 80, 83, 86, 86, 88, 89, 86, 81, 82, 80, 81, 83, 79, 80, 75, 80, 81, 84, 87, 90, 85, 83, 82, 78, 79, 79, 87, 89, 88, 84, 85, 85, 88, 88, 91, 87, 91, 90, 93, 94, 92, 93, 92, 91, 91, 90, 90, 86, 86, 84, 83, 81, 81, 88, 89, 86, 86, 89, 91, 96, 101, 93, 89, 89, 86, 88, 87, 90, 90, 81, 88, 91, 95, 96, 94, 93, 92, 86, 90, 92, 93, 91, 87, 90, 83, 84, 88, 90, 84, 83, 85, 85, 83, 85, 89, 93, 94, 95, 98, 98, 97, 101, 99, 97, 99, 97, 99, 101, 102, 102, 100, 100, 102, 101, 100, 99, 100, 96, 99, 101, 96, 97, 96, 95, 93, 92, 95, 96, 97, 97, 100, 98, 98, 98, 101, 99, 100, 102, 100, 97, 96, 95, 93, 95, 94, 94, 94, 97, 100, 98, 100, 103, 103, 104, 100, 98, 99, 98, 98, 94, 96, 97, 102, 102, 99, 99, 100, 100, 101, 103, 96, 99, 97, 97, 94, 95, 93, 92, 90, 91, 92, 94, 98, 94, 93, 91, 93, 94, 92, 93, 97, 96, 96, 98, 100, 100, 99, 100, 99, 107, 104, 103, 101, 102, 101, 102, 104, 105, 106, 101, 102, 100, 101, 101, 99, 100, 100, 104, 108, 106, 102, 100, 101, 101, 104, 101, 101, 104, 107, 108, 109, 109, 111, 110, 111, 111, 111, 110, 111, 116, 116, 114, 114, 115, 113, 114, 117, 119, 119, 117, 114, 115, 119, 119, 119, 121, 122, 122, 123, 119, 119, 118, 115, 114, 114, 115, 114, 116, 113, 114, 115, 114, 114, 116, 118, 117, 118, 118, 122, 122, 123, 125, 116, 115, 114, 114, 115, 114, 112, 116, 115, 115, 114, 114, 118, 118, 116, 110, 101, 99, 98, 96, 96, 96, 96, 98, 95, 96, 96, 99, 102, 100, 101, 96, 94, 98, 99, 96, 95, 97, 94, 97, 98, 99, 100, 101, 100, 100, 101, 101, 101, 98, 98, 99, 100, 101, 103, 104, 105, 103, 103, 103, 98, 101, 99, 102, 103, 102, 99, 100, 101, 95, 96, 96, 97, 97, 98, 99, 98, 98, 100, 97, 95, 95, 95, 95, 95, 95, 98, 97, 98, 102, 98, 100, 98, 97, 96, 100, 102, 99, 100, 103, 103, 110, 111, 106, 100, 99, 100, 99, 99, 99, 103, 104, 99, 101, 100, 98, 94, 94, 95, 97, 98, 102, 102, 103, 101, 103, 101, 102, 98, 96, 96, 94, 96, 95, 98, 97, 96, 98, 98, 101, 101, 97, 94, 97, 100, 100, 101, 109, 106, 107, 113, 111, 109, 107, 107, 111, 112, 105, 107, 111, 111, 112, 108, 104, 107, 110, 109, 110, 110, 112, 110, 106, 110, 109, 108, 115, 120, 116, 114, 119, 121, 122, 123, 119, 120, 108, 110, 108, 112, 115, 117, 114, 112, 107, 108, 109, 112, 114, 113, 109, 107, 106, 103, 102, 102, 102, 105, 102, 101, 100, 105, 105, 102, 103, 103, 101, 101, 99, 102, 99, 98, 97, 101, 101, 104, 105, 106, 69, 63, 59, 58, 63, 61, 59, 59, 58, 59, 62, 66, 68, 68, 69, 64, 64, 67, 63, 64, 57, 56, 57, 59, 58, 56, 53, 54, 52, 52, 59, 56, 60, 60, 61, 60, 60, 61, 61, 60, 58, 55, 54, 55, 55, 54, 63, 60, 61, 58, 65, 64, 63, 66, 61, 59, 64, 68, 67, 64, 63, 60, 57, 56, 57, 57, 54, 57, 56, 59, 61, 56, 52, 52, 51, 51, 49, 49, 49, 49, 43, 43, 44, 43, 42, 44, 46, 45, 45, 53, 51, 47, 49, 48, 49, 49, 49, 46, 45, 49, 50, 50, 55, 59, 50, 53, 51, 53, 52, 51, 55, 51, 49, 51, 50, 51, 54, 53, 58, 59, 59, 56, 54, 54, 54, 55, 51, 50, 49, 50, 52, 49, 53, 45, 47, 44, 44, 45, 50, 53, 54, 50, 50, 51, 48, 47, 50, 50, 44, 48, 49, 43, 45, 43, 40, 41, 41, 41, 41, 40, 48, 50, 50, 53, 54, 54, 56, 57, 61, 62, 61, 64, 67, 67, 68, 69, 72, 80, 83, 85, 86, 89, 93, 96, 99, 104, 109, 111, 113, 111, 115, 121, 124, 127, 130, 132, 133, 131, 132, 135, 136, 137, 137, 138, 140, 144, 153, 156, 156, 158, 159, 122, 52, 48, 48, 53, 62, 62, 58, 58, 56, 53, 56, 56, 56, 57, 54, 55, 55, 56, 59, 59, 56, 55, 54, 53, 53, 57, 54, 58, 59, 61, 64, 59, 59, 60, 58, 56, 60, 65, 63, 63, 58, 57, 62, 65, 64, 65, 67, 70, 70, 69, 69, 71, 69, 67, 66, 66, 67, 68, 65, 70, 67, 70, 71, 75, 73, 73, 75, 78, 76, 74, 74, 74, 74, 77, 75, 73, 78, 78, 74, 75, 75, 74, 73, 74, 74, 77, 78, 78, 78, 79, 83, 84, 86, 86, 90, 90, 91, 92, 93, 93, 93, 95, 87, 87, 85, 90, 93, 92, 91, 88, 85, 89, 90, 90, 93, 91, 84, 85, 84, 85, 86, 87, 89, 88, 91, 91, 91, 91, 87, 85, 85, 86, 88, 82, 85, 87, 88, 86, 87, 89, 88, 84, 85, 84, 87, 91, 92, 93, 93, 93, 89, 90, 89, 89, 93, 92, 94, 92, 92, 94, 85, 84, 87, 86, 87, 86, 84, 82, 85, 86, 83, 83, 85, 88, 86, 87, 85, 87, 88, 92, 85, 87, 89, 90, 92, 91, 89, 89, 90, 90, 90, 92, 94, 94, 95, 95, 98, 101, 99, 99, 105, 110, 109, 104, 104, 108, 115, 115, 115, 116, 115, 116, 112, 110, 108, 113, 113, 107, 106, 106, 109, 108, 108, 107, 107, 105, 105, 109, 111, 110, 109, 106, 105, 106, 106, 100, 100, 103, 103, 107, 107, 109, 110, 108, 109, 107, 102, 102, 100, 97, 98, 101, 104, 107, 107, 110, 111, 101, 102, 106, 103, 101, 105, 109, 109, 112, 110, 115, 114, 113, 104, 103, 105, 108, 108, 109, 106, 108, 111, 110, 104, 108, 108, 107, 112, 109, 113, 112, 111, 115, 118, 117, 114, 119, 122, 119, 122, 118, 118, 119, 118, 115, 114, 116, 119, 122, 121, 118, 118, 122, 125, 122, 122, 117, 118, 117, 118, 118, 121, 121, 124, 124, 125, 126, 125, 130, 131, 132, 126, 124, 127, 127, 129, 131, 132, 133, 134, 136, 131, 132, 133, 131, 129, 128, 129, 128, 127, 130, 128, 127, 127, 129, 131, 130, 131, 130, 135, 136, 136, 137, 140, 139, 134, 136, 135, 139, 140, 137, 129, 130, 129, 133, 131, 130, 127, 128, 129, 129, 129, 129, 130, 129, 130, 134, 138, 140, 141, 141, 140, 143, 141, 138, 140, 139, 140, 146, 145, 144, 143, 144, 145, 144, 141, 141, 143, 145, 146, 143, 137, 140, 136, 137, 135, 134, 139, 140, 145, 147, 148, 146, 149, 150, 153, 146, 144, 142, 143, 143, 144, 146, 144, 142, 146, 148, 149, 147, 144, 150, 150, 147, 148, 147, 147, 145, 145, 142, 142, 145, 146, 147, 142, 145, 140, 143, 144, 143, 148, 145, 148, 147, 149, 149, 154, 155, 157, 156, 150, 150, 151, 155, 156, 152, 152, 153, 156, 157, 155, 153, 153, 157, 151, 150, 149, 148, 145, 144, 141, 145, 146, 147, 151, 152, 150, 147, 151, 149, 148, 150, 147, 148, 149, 155, 159, 154, 151, 148, 149, 148, 149, 154, 157, 158, 145, 143, 143, 140, 144, 145, 146, 146, 148, 145, 146, 149, 150, 151, 148, 145, 146, 148, 150, 149, 154, 151, 150, 147, 151, 156, 158, 154, 151, 152, 149, 154, 156, 158, 162, 165, 166, 165, 159, 161, 163, 159, 151, 154, 157, 156, 156, 163, 161, 160, 156, 155, 155, 156, 158, 155, 157, 153, 160, 160, 158, 151, 151, 153, 153, 155, 152, 151, 152, 151, 151, 155, 157, 158, 161, 164, 166, 168, 174, 174, 174, 176, 180, 182, 185, 183, 183, 187, 192, 194, 193, 191, 192, 163, 153, 152, 156, 158, 158, 155, 156, 157, 159, 162, 161, 160, 155, 156, 159, 160, 166, 168, 168, 161, 164, 165, 168, 169, 169, 172, 169, 169, 167, 166, 166, 167, 164, 163, 160, 161, 159, 161, 163, 161, 161, 154, 156, 157, 160, 156, 158, 162, 163, 163, 166, 165, 166, 168, 169, 169, 166, 168, 170, 170, 171, 170, 168, 172, 173, 176, 175, 175, 172, 171, 170, 172, 174, 174, 171, 172, 173, 172, 172, 178, 182, 185, 186, 182, 181, 180, 182, 179, 182, 183, 185, 185, 189, 193, 190, 195, 194, 193, 191, 193, 194, 194, 192, 191, 192, 194, 192, 192, 194, 196, 194, 199, 197, 200, 205, 199, 200, 201, 202, 203, 207, 210, 211, 213, 211, 213, 213, 215, 220, 220, 218, 215, 215, 218, 220, 225, 223, 220, 224, 223, 225, 227, 227, 229, 227, 228, 228, 229, 229, 226, 228, 226, 227, 223, 226, 226, 233, 232, 229, 231, 232, 230, 231, 236, 240, 239, 239, 239, 242, 243, 247, 246, 249, 251, 252, 252, 255, 255, 258, 261, 261, 262, 264, 268, 270, 261, 264, 264, 264, 264, 267, 264, 260, 258, 251, 245, 245, 249, 253, 251, 248, 246, 240, 241, 245, 250, 251, 248, 253, 244, 247, 250, 253, 251, 242, 240, 230, 237, 236, 238, 239, 234, 229, 229, 229, 228, 230, 228, 225, 233, 235, 238, 237, 237, 230, 231, 233, 235, 236, 235, 232, 230, 231, 223, 224, 222, 222, 222, 224, 224, 228, 226, 226, 224, 228, 224, 223, 223, 225, 228, 228, 230, 230, 230, 235, 237, 238, 244, 239, 245, 245, 246, 244, 246, 248, 248, 247, 245, 245, 249, 248, 253, 248, 248, 249, 248, 253, 257, 258, 255, 257, 257, 256, 256, 256, 257, 259, 257, 256, 260, 260, 261, 262, 262, 266, 264, 267, 266, 263, 268, 263, 261, 265, 264, 263, 262, 264, 264, 264, 262, 263, 265, 268, 274, 268, 268, 268, 273, 269, 273, 277, 280, 280, 282, 282, 282, 284, 286, 283, 286, 284, 280, 279, 283, 282, 283, 281, 279, 283, 290, 288, 287, 285, 287, 286, 285, 289, 293, 293, 274, 269, 267, 265, 268, 273, 275, 277, 277, 276, 278, 280, 284, 279, 279, 282, 285, 289, 290, 289, 287, 285, 288, 289, 289, 293, 295, 296, 300, 297, 294, 292, 296, 295, 287, 286, 292, 289, 282, 283, 285, 278, 277, 275, 272, 271, 272, 271, 256, 255, 241, 238, 234, 232, 230, 222, 215, 214, 209, 203, 205, 206, 200, 201, 194, 199, 195, 199, 201, 190, 163, 166, 163, 155, 161, 162, 155, 145, 147, 142, 144, 145, 147, 145, 140, 141, 134, 139, 144, 140, 136, 143, 146, 146, 147, 148, 146, 151, 154, 147, 149, 151, 157, 160, 158, 158, 157, 157, 159, 156, 157, 155, 150, 155, 157, 158, 161, 162, 159, 160, 159, 160, 162, 158, 156, 160, 158, 155, 156, 156, 161, 160, 165, 167, 168, 166, 166, 167, 168, 165, 168, 168, 170, 174, 174, 173, 170, 165, 165, 165, 166, 173, 172, 168, 168, 169, 152, 154, 155, 162, 161, 166, 167, 169, 172, 172, 168, 167, 171, 172, 176, 178, 180, 180, 179, 180, 184, 184, 185, 180, 176, 179, 181, 172, 171, 174, 179, 186, 184, 174, 176, 176, 179, 181, 181, 179, 180, 182, 184, 184, 184, 188, 188, 187, 184, 186, 185, 187, 187, 188, 190, 191, 191, 194, 192, 190, 189, 186, 191, 189, 189, 195, 199, 198, 197, 198, 201, 203, 207, 203, 202, 200, 199, 196, 201, 200, 203, 201, 203, 203, 201, 203, 202, 199, 197, 197, 197, 198, 198, 197, 194, 199, 201, 199, 198, 199, 205, 198, 196, 204, 207, 209, 210, 205, 209, 208, 209, 211, 212, 217, 217, 218, 218, 213, 221, 220, 219, 216, 213, 214, 211, 209, 212, 213, 215, 213, 211, 210, 207, 209, 209, 212, 208, 208, 206, 206, 208, 209, 212, 210, 210, 212, 216, 217, 219, 225, 218, 217, 217, 215, 218, 219, 220, 219, 217, 219, 214, 209, 209, 213, 211, 218, 220, 220, 224, 222, 224, 222, 230, 231, 225, 224, 226, 237, 236, 233, 235, 232, 235, 232, 232, 232, 229, 231, 230, 231, 231, 235, 231, 224, 226, 224, 223, 223, 228, 224, 225, 221, 208, 210, 209, 210, 209, 211, 213, 211, 208, 218, 216, 215, 215, 214, 213, 216, 221, 224, 225, 226, 233, 231, 231, 232, 226, 229, 227, 225, 223, 220, 222, 226, 229, 230, 231, 230, 226, 230, 230, 230, 231, 237, 235, 234, 234, 237, 198, 198, 198, 198, 196, 199, 203, 204, 197, 200, 198, 199, 200, 199, 197, 191, 190, 177, 185, 187, 188, 183, 197, 196, 192, 185, 189, 191, 187, 184, 186, 187, 189, 189, 189, 191, 186, 191, 191, 194, 195, 196, 199, 201, 204, 205, 199, 201, 202, 201, 197, 202, 203, 199, 201, 204, 206, 206, 200, 203, 202, 200, 202, 202, 202, 206, 201, 203, 202, 203, 201, 198, 200, 201, 203, 206, 206, 199, 199, 200, 201, 201, 202, 201, 203, 204, 192, 197, 198, 197, 197, 197, 201, 201, 203, 205, 207, 206, 211, 210, 210, 211, 215, 219, 221, 214, 216, 218, 217, 216, 222, 221, 220, 223, 219, 220, 220, 226, 226, 225, 228, 225, 220, 221, 220, 219, 223, 224, 228, 228, 226, 225, 206, 202, 202, 209, 215, 219, 228, 225, 223, 226, 229, 233, 234, 234, 242, 245, 248, 254, 264, 267, 279, 285, 285, 289, 294, 296, 300, 306, 306, 311, 317, 325, 327, 333, 339, 339, 344, 346, 344, 348, 352, 360, 363, 363, 370, 375, 379, 382, 387, 393, 398, 400, 403, 402, 406, 408, 201, 204, 209, 211, 213, 218, 219, 219, 219, 217, 211, 212, 218, 220, 218, 215, 220, 217, 220, 224, 224, 225, 223, 226, 222, 228, 229, 233, 232, 233, 234, 237, 243, 242, 242, 241, 239, 235, 237, 236, 241, 246, 255, 255, 258, 254, 255, 256, 262, 259, 250, 249, 240, 238, 240, 224, 230, 234, 235, 232, 228, 226, 224, 219, 217, 219, 214, 212, 212, 207, 212, 213, 218, 223, 220, 221, 226, 224, 226, 226, 235, 236, 237, 244, 239, 238, 237, 239, 242, 246, 253, 259, 259, 248, 246, 250, 250, 248, 245, 242, 238, 243, 242, 238, 238, 239, 241, 241, 245, 244, 248, 246, 243, 246, 245, 253, 252, 253, 259, 260, 264, 258, 251, 247, 248, 248, 247, 246, 250, 254, 257, 256, 254, 252, 252, 247, 241, 243, 242, 246, 251, 253, 254, 256, 259, 258, 260, 259, 259, 262, 260, 264, 263, 263, 264, 271, 271, 268, 272, 274, 273, 277, 281, 282, 281, 275, 277, 278, 271, 269, 272, 273, 272, 274, 276, 274, 272, 275, 271, 272, 272, 271, 278, 278, 278, 275, 272, 271, 277, 269, 267, 268, 264, 263, 264, 271, 274, 272, 275, 272, 270, 265, 267, 263, 263, 266, 265, 265, 268, 271, 276, 276, 284, 285, 286, 287, 291, 292, 290, 296, 297, 295, 297, 299, 300, 296, 294, 291, 293, 299, 308, 300, 300, 301, 304, 305, 303, 306, 301, 299, 297, 298, 299, 289, 295, 300, 300, 303, 307, 307, 307, 310, 312, 310, 311, 309, 303, 302, 307, 309, 312, 315, 312, 311, 312, 311, 316, 313, 318, 319, 325, 328, 327, 321, 323, 323, 325, 320, 325, 330, 332, 339, 340, 341, 337, 337, 337, 336, 339, 344, 349, 348, 345, 344, 342, 349, 345, 336, 340, 340, 338, 345, 346, 349, 344, 341, 344, 345, 349, 348, 352, 359, 359, 359, 350, 357, 358, 358, 358, 360, 361, 366, 366, 367, 367, 368, 371, 367, 374, 373, 366, 367, 368, 366, 370, 373, 373, 378, 379, 382, 385, 385, 385, 381, 379, 378, 373, 379, 373, 376, 370, 368, 359, 359, 366, 367, 369, 368, 367, 366, 353, 345, 345, 356, 352, 351, 353, 354, 354, 353, 350, 348, 347, 342, 341, 342, 348, 346, 348, 350, 349, 344, 351, 352, 356, 356, 355, 358, 358, 357, 360, 361, 361, 360, 357, 355, 355, 356, 361, 358, 357, 350, 348, 349, 350, 351, 352, 353, 342, 344, 348, 351, 350, 348, 347, 349, 352, 354, 352, 348, 354, 359, 358, 357, 356, 359, 360, 355, 354, 353, 358, 361, 365, 367, 371, 368, 368, 366, 362, 361, 362, 365, 364, 360, 358, 360, 360, 358, 357, 359, 362, 364, 367, 373, 381, 386, 369, 365, 365, 366, 368, 372, 375, 376, 379, 377, 382, 383, 392, 383, 380, 378, 378, 377, 376, 382, 383, 382, 382, 381, 386, 386, 386, 384, 385, 389, 387, 387, 383, 389, 393, 390, 392, 396, 395, 394, 397, 399, 398, 390, 387, 391, 397, 398, 400, 399, 399, 393, 391, 387, 387, 388, 385, 386, 383, 386, 382, 385, 385, 395, 395, 395, 392, 397, 394, 392, 391, 393, 396, 397, 397, 399, 395, 396, 399, 399, 398, 401, 406, 407, 407, 406, 407, 413, 418, 412, 415, 410, 408, 404, 416, 416, 413, 415, 420, 423, 414, 417, 422, 428, 428, 432, 432, 428, 431, 440, 445, 445, 443, 436, 436, 438, 437, 437, 438, 436, 435, 436, 435, 438, 437, 429, 424, 424, 422, 425, 427, 434, 436, 431, 434, 438, 441, 443, 450, 449, 449, 452, 455, 449, 449, 455, 453, 452, 457, 454, 456, 457, 461, 461, 455, 446, 454, 461, 459, 463, 455, 457, 454, 449, 455, 456, 455, 463, 462, 467, 477, 476, 471, 473, 472, 470, 468, 467, 465, 468, 467, 464, 464, 465, 471, 470, 476, 478, 472, 474, 473, 472, 472, 471, 474, 483, 489, 486, 484, 482, 485, 490, 490, 487, 486, 482, 485, 489, 491, 491, 492, 485, 482, 486, 481, 483, 488, 491, 487, 480, 480, 484, 480, 478, 481, 485, 486, 488, 486, 489, 491, 497, 499, 497, 482, 478, 484, 484, 487, 481, 475, 478, 476, 473, 475, 475, 470, 472, 474, 467, 472, 474, 477, 480, 483, 483, 485, 489, 487, 487, 489, 491, 491, 491, 495, 498, 496, 496, 496, 496, 498, 499, 498, 501, 503, 505, 505, 507, 506, 507, 504, 498, 499, 500, 506, 510, 508, 502, 509, 506, 506, 510, 510, 513, 510, 512, 510, 510, 514, 514, 514, 509, 503, 501, 506, 505, 509, 515, 513, 511, 514, 510, 513, 513, 516, 521, 521, 520, 523, 526, 533, 527, 519, 519, 521, 526, 528, 533, 526, 527, 521, 518, 519, 522, 520, 520, 512, 503, 501, 494, 492, 491, 488, 489, 486, 472, 470, 471, 471, 472, 478, 478, 476, 475, 473, 476, 477, 475, 477, 471, 461, 462, 464, 467, 465, 455, 444, 445, 446, 443, 441, 435, 423, 418, 418, 414, 418, 423, 416, 406, 405, 397, 392, 394, 391, 390, 388, 384, 373, 374, 371, 367, 370, 365, 364, 355, 355, 352, 355, 354, 361, 364, 365, 366, 361, 357, 363, 369, 366, 358, 361, 363, 359, 357, 352, 348, 347, 346, 346, 347, 337, 340, 336, 336, 335, 334, 333, 332, 331, 328, 330, 334, 335, 337, 337, 343, 343, 338, 341, 345, 340, 336, 333, 331, 324, 328, 337, 337, 332, 332, 333, 338, 341, 343, 346, 341, 345, 345, 348, 348, 352, 343, 343, 337, 342, 343, 352, 354, 353, 346, 347, 347, 355, 355, 355, 351, 351, 351, 357, 357, 358, 360, 351, 352, 356, 356, 353, 350, 356, 354, 360, 353, 354, 350, 351, 350, 355, 351, 349, 353, 354, 360, 364, 360, 353, 356, 348, 347, 348, 349, 346, 347, 349, 352, 354, 355, 362, 357, 359, 356, 355, 352, 356, 356, 354, 353, 356, 361, 358, 359, 361, 368, 370, 367, 374, 374, 367, 370, 372, 374, 375, 378, 380, 383, 384, 376, 376, 375, 375, 380, 384, 385, 378, 372, 371, 376, 374, 375, 384, 377, 377, 368, 369, 371, 372, 377, 373, 374, 371, 370, 372, 376, 376, 378, 376, 377, 377, 379, 380, 383, 385, 382, 382, 375, 377, 383, 379, 373, 379, 378, 384, 386, 391, 395, 396, 392, 393, 397, 393, 397, 394, 389, 382, 380, 390, 388, 390, 391, 388, 381, 376, 377, 380, 384, 382, 381, 381, 381, 380, 378, 380, 381, 386, 387, 385, 387, 385, 390, 392, 392, 380, 380, 380, 380, 379, 389, 396, 391, 388, 387, 393, 399, 396, 401, 398, 389, 389, 393, 389, 393, 395, 395, 394, 396, 404, 405, 397, 399, 400, 392, 388, 387, 387, 391, 399, 398, 402, 398, 407, 406, 409, 405, 406, 405, 402, 399, 402, 402, 408, 405, 402, 402, 401, 405, 404, 401, 395, 396, 401, 395, 396, 398, 403, 402, 404, 408, 411, 409, 411, 408, 410, 412, 416, 419, 413, 412, 407, 406, 405, 404, 400, 402, 402, 402, 407, 411, 401, 403, 407, 407, 406, 404, 401, 398, 400, 409, 406, 408, 408, 406, 412, 413, 410, 413, 410, 405, 403, 400, 397, 397, 401, 395, 395, 395, 396, 401, 403, 406, 404, 402, 404, 403, 401, 402, 408, 404, 409, 413, 417, 422, 422, 428, 427, 424, 421, 423, 423, 424, 426, 421, 423, 418, 419, 421, 428, 429, 424, 414, 411, 409, 410, 411, 408, 413, 411, 409, 407, 401, 401, 396, 393, 398, 396, 394, 394, 401, 403, 403, 400, 400, 404, 405, 409, 413, 404, 409, 412, 415, 418, 421, 427, 424, 422, 418, 411, 413, 415, 419, 417, 417, 414, 413, 415, 419, 423, 419, 418, 416, 419, 418, 420, 421, 421, 421, 420, 419, 421, 423, 432, 429, 429, 428, 428, 429, 440, 442, 436, 438, 437, 439, 428, 435, 430, 430, 430, 428, 430, 435, 435, 440, 442, 446, 446, 444, 449, 453, 452, 445, 296, 253, 220, 226], "yaxis": "y"}, {"hovertemplate": "<b>OLS trendline</b><br>workload = 0.0987386 * day_offset + -11.2727<br>R<sup>2</sup>=0.845679<br><br>day_offset=%{x}<br>workload=%{y} <b>(trend)</b><extra></extra>", "legendgroup": "", "marker": {"color": "#636efa", "symbol": "circle"}, "mode": "lines", "name": "", "showlegend": false, "type": "scatter", "x": [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 101, 102, 103, 104, 105, 106, 107, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 179, 180, 181, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 309, 311, 313, 314, 315, 316, 318, 319, 321, 322, 323, 324, 325, 326, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 339, 340, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 383, 384, 385, 386, 387, 388, 390, 391, 392, 393, 395, 396, 397, 398, 399, 400, 401, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 430, 431, 433, 434, 435, 436, 438, 439, 440, 442, 443, 444, 445, 446, 447, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 461, 462, 463, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 661, 662, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 860, 861, 862, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 892, 893, 894, 895, 896, 897, 898, 899, 901, 902, 903, 904, 905, 907, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 972, 973, 974, 975, 976, 978, 979, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1082, 1084, 1085, 1086, 1087, 1088, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1112, 1113, 1114, 1115, 1116, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240, 1241, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249, 1250, 1252, 1253, 1254, 1255, 1257, 1258, 1259, 1260, 1261, 1262, 1263, 1264, 1265, 1266, 1267, 1268, 1269, 1270, 1271, 1272, 1273, 1274, 1275, 1276, 1277, 1279, 1280, 1281, 1282, 1283, 1284, 1285, 1286, 1287, 1288, 1289, 1290, 1291, 1292, 1293, 1294, 1295, 1296, 1297, 1298, 1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309, 1310, 1311, 1312, 1315, 1316, 1317, 1318, 1319, 1320, 1321, 1322, 1323, 1324, 1325, 1326, 1328, 1329, 1330, 1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341, 1342, 1343, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1359, 1360, 1361, 1362, 1363, 1364, 1365, 1366, 1367, 1368, 1369, 1370, 1371, 1372, 1373, 1374, 1375, 1376, 1378, 1379, 1380, 1381, 1382, 1383, 1384, 1385, 1386, 1387, 1388, 1389, 1391, 1392, 1393, 1394, 1395, 1396, 1397, 1398, 1399, 1400, 1401, 1402, 1403, 1404, 1406, 1407, 1408, 1409, 1410, 1412, 1413, 1414, 1415, 1416, 1417, 1418, 1419, 1420, 1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428, 1429, 1430, 1431, 1432, 1433, 1434, 1435, 1436, 1437, 1438, 1439, 1440, 1441, 1442, 1443, 1444, 1445, 1446, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456, 1457, 1458, 1459, 1460, 1461, 1462, 1463, 1464, 1465, 1466, 1467, 1468, 1469, 1470, 1471, 1472, 1473, 1474, 1475, 1476, 1477, 1478, 1479, 1480, 1482, 1483, 1484, 1485, 1486, 1487, 1488, 1490, 1491, 1492, 1493, 1494, 1495, 1496, 1497, 1498, 1499, 1500, 1501, 1502, 1503, 1504, 1505, 1506, 1507, 1508, 1509, 1510, 1511, 1512, 1513, 1514, 1515, 1516, 1517, 1518, 1519, 1520, 1521, 1522, 1523, 1524, 1525, 1526, 1527, 1528, 1529, 1530, 1531, 1532, 1533, 1534, 1535, 1536, 1537, 1538, 1539, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549, 1550, 1551, 1552, 1553, 1554, 1555, 1556, 1557, 1558, 1560, 1561, 1562, 1563, 1564, 1565, 1566, 1567, 1568, 1569, 1570, 1571, 1572, 1573, 1574, 1575, 1576, 1577, 1578, 1579, 1580, 1581, 1582, 1583, 1584, 1585, 1586, 1587, 1588, 1589, 1591, 1592, 1593, 1594, 1595, 1596, 1597, 1600, 1601, 1602, 1603, 1605, 1606, 1607, 1608, 1609, 1610, 1611, 1612, 1613, 1614, 1615, 1616, 1617, 1618, 1619, 1620, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 1635, 1636, 1637, 1638, 1639, 1640, 1641, 1642, 1643, 1644, 1645, 1646, 1647, 1648, 1649, 1650, 1651, 1652, 1653, 1654, 1655, 1656, 1657, 1658, 1659, 1660, 1661, 1662, 1663, 1664, 1665, 1666, 1667, 1668, 1669, 1670, 1671, 1672, 1673, 1674, 1675, 1676, 1677, 1679, 1680, 1681, 1682, 1683, 1684, 1685, 1686, 1687, 1688, 1689, 1690, 1691, 1692, 1693, 1694, 1695, 1696, 1697, 1698, 1699, 1700, 1701, 1702, 1703, 1704, 1705, 1706, 1707, 1708, 1709, 1710, 1711, 1712, 1713, 1714, 1715, 1716, 1717, 1718, 1719, 1720, 1721, 1722, 1723, 1724, 1725, 1726, 1727, 1728, 1729, 1730, 1731, 1732, 1733, 1735, 1736, 1737, 1738, 1739, 1740, 1741, 1742, 1743, 1744, 1745, 1746, 1747, 1748, 1749, 1750, 1751, 1752, 1753, 1754, 1755, 1756, 1757, 1758, 1759, 1760, 1761, 1762, 1763, 1764, 1765, 1766, 1767, 1768, 1769, 1770, 1771, 1772, 1773, 1774, 1775, 1776, 1777, 1778, 1779, 1780, 1781, 1782, 1783, 1784, 1785, 1786, 1787, 1788, 1789, 1790, 1791, 1792, 1793, 1794, 1795, 1796, 1797, 1798, 1799, 1800, 1801, 1802, 1803, 1804, 1805, 1806, 1807, 1808, 1809, 1810, 1811, 1812, 1813, 1814, 1815, 1816, 1817, 1818, 1819, 1820, 1821, 1822, 1823, 1824, 1825, 1826, 1827, 1828, 1829, 1830, 1831, 1832, 1833, 1834, 1835, 1836, 1837, 1838, 1839, 1840, 1841, 1842, 1843, 1844, 1845, 1847, 1848, 1849, 1850, 1851, 1852, 1853, 1854, 1855, 1856, 1857, 1858, 1859, 1860, 1861, 1862, 1863, 1864, 1865, 1866, 1867, 1868, 1869, 1870, 1871, 1872, 1873, 1874, 1875, 1876, 1877, 1878, 1879, 1880, 1881, 1882, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 1890, 1891, 1892, 1893, 1894, 1895, 1896, 1897, 1898, 1899, 1900, 1901, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910, 1911, 1912, 1913, 1914, 1915, 1916, 1917, 1918, 1919, 1920, 1921, 1922, 1923, 1924, 1925, 1926, 1927, 1928, 1929, 1930, 1931, 1932, 1933, 1934, 1935, 1936, 1937, 1938, 1939, 1940, 1941, 1942, 1943, 1944, 1945, 1946, 1947, 1948, 1949, 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 2058, 2059, 2060, 2061, 2062, 2063, 2064, 2065, 2066, 2067, 2068, 2069, 2070, 2071, 2072, 2073, 2074, 2075, 2076, 2077, 2078, 2079, 2080, 2081, 2082, 2083, 2084, 2085, 2086, 2087, 2088, 2089, 2090, 2091, 2092, 2093, 2094, 2095, 2096, 2097, 2098, 2099, 2100, 2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112, 2113, 2114, 2115, 2116, 2117, 2118, 2119, 2120, 2121, 2122, 2123, 2124, 2125, 2126, 2127, 2128, 2129, 2130, 2131, 2132, 2133, 2134, 2135, 2136, 2137, 2138, 2139, 2140, 2141, 2142, 2143, 2144, 2145, 2146, 2147, 2148, 2149, 2150, 2151, 2152, 2153, 2154, 2155, 2156, 2157, 2158, 2159, 2160, 2161, 2162, 2163, 2164, 2165, 2166, 2167, 2168, 2169, 2170, 2171, 2172, 2173, 2174, 2175, 2176, 2177, 2178, 2179, 2180, 2181, 2182, 2183, 2184, 2185, 2186, 2187, 2188, 2189, 2190, 2191, 2192, 2193, 2194, 2195, 2196, 2197, 2198, 2199, 2200, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212, 2213, 2214, 2215, 2216, 2217, 2218, 2219, 2220, 2221, 2222, 2223, 2224, 2225, 2226, 2227, 2228, 2229, 2230, 2231, 2232, 2233, 2234, 2235, 2236, 2237, 2238, 2239, 2240, 2241, 2242, 2243, 2244, 2245, 2246, 2247, 2248, 2249, 2250, 2251, 2252, 2253, 2254, 2255, 2256, 2257, 2258, 2259, 2260, 2261, 2262, 2263, 2264, 2265, 2266, 2267, 2268, 2269, 2270, 2271, 2272, 2273, 2274, 2275, 2276, 2277, 2278, 2279, 2280, 2281, 2282, 2283, 2284, 2285, 2286, 2287, 2288, 2289, 2290, 2291, 2292, 2293, 2294, 2295, 2296, 2297, 2298, 2299, 2300, 2301, 2302, 2303, 2304, 2305, 2306, 2307, 2308, 2309, 2310, 2311, 2312, 2313, 2314, 2315, 2316, 2317, 2318, 2319, 2320, 2321, 2322, 2323, 2324, 2325, 2326, 2327, 2328, 2329, 2330, 2331, 2332, 2333, 2334, 2335, 2336, 2337, 2338, 2339, 2341, 2342, 2343, 2344, 2345, 2346, 2347, 2348, 2349, 2350, 2351, 2352, 2353, 2354, 2355, 2356, 2357, 2358, 2359, 2360, 2361, 2362, 2363, 2364, 2365, 2366, 2367, 2368, 2369, 2370, 2371, 2372, 2373, 2374, 2375, 2376, 2377, 2378, 2379, 2380, 2381, 2382, 2383, 2384, 2385, 2386, 2387, 2388, 2389, 2390, 2391, 2392, 2393, 2394, 2395, 2396, 2397, 2398, 2399, 2400, 2401, 2402, 2403, 2404, 2405, 2406, 2407, 2408, 2409, 2410, 2411, 2412, 2413, 2414, 2415, 2416, 2417, 2418, 2419, 2420, 2421, 2422, 2423, 2424, 2425, 2426, 2427, 2429, 2430, 2431, 2432, 2433, 2434, 2435, 2436, 2437, 2438, 2439, 2440, 2441, 2442, 2443, 2444, 2445, 2446, 2447, 2448, 2449, 2450, 2451, 2452, 2453, 2454, 2455, 2456, 2457, 2458, 2459, 2460, 2461, 2462, 2463, 2464, 2465, 2466, 2467, 2468, 2469, 2470, 2471, 2472, 2473, 2474, 2475, 2476, 2477, 2478, 2479, 2480, 2481, 2482, 2483, 2484, 2485, 2486, 2487, 2488, 2489, 2490, 2491, 2492, 2493, 2494, 2495, 2496, 2497, 2498, 2499, 2500, 2501, 2502, 2503, 2504, 2505, 2506, 2507, 2508, 2509, 2510, 2511, 2512, 2513, 2514, 2515, 2516, 2517, 2518, 2519, 2520, 2521, 2522, 2523, 2524, 2525, 2526, 2527, 2528, 2529, 2530, 2531, 2532, 2533, 2534, 2535, 2536, 2537, 2539, 2540, 2541, 2542, 2543, 2544, 2545, 2546, 2547, 2548, 2549, 2550, 2551, 2552, 2553, 2554, 2555, 2556, 2557, 2558, 2559, 2560, 2561, 2562, 2563, 2564, 2565, 2566, 2567, 2568, 2569, 2570, 2571, 2572, 2573, 2574, 2575, 2576, 2577, 2578, 2579, 2580, 2581, 2582, 2583, 2584, 2585, 2586, 2587, 2588, 2589, 2590, 2591, 2592, 2593, 2594, 2595, 2596, 2597, 2598, 2599, 2600, 2601, 2602, 2603, 2604, 2605, 2606, 2607, 2608, 2609, 2610, 2611, 2612, 2613, 2614, 2615, 2616, 2617, 2618, 2619, 2620, 2621, 2622, 2623, 2624, 2625, 2626, 2627, 2628, 2629, 2630, 2631, 2632, 2633, 2634, 2635, 2636, 2637, 2638, 2639, 2640, 2641, 2642, 2643, 2644, 2645, 2646, 2647, 2648, 2649, 2650, 2651, 2652, 2653, 2654, 2655, 2656, 2657, 2658, 2659, 2660, 2661, 2662, 2663, 2664, 2665, 2666, 2667, 2668, 2669, 2670, 2671, 2672, 2673, 2674, 2675, 2676, 2677, 2678, 2679, 2680, 2681, 2682, 2683, 2684, 2685, 2686, 2687, 2688, 2689, 2690, 2691, 2692, 2693, 2694, 2695, 2696, 2697, 2698, 2699, 2700, 2701, 2702, 2703, 2704, 2705, 2706, 2707, 2708, 2709, 2710, 2711, 2712, 2713, 2714, 2715, 2716, 2717, 2718, 2719, 2720, 2721, 2722, 2723, 2724, 2725, 2726, 2727, 2728, 2729, 2730, 2731, 2732, 2733, 2734, 2735, 2736, 2737, 2738, 2739, 2740, 2741, 2742, 2743, 2744, 2745, 2746, 2747, 2748, 2749, 2750, 2751, 2752, 2753, 2754, 2755, 2756, 2757, 2758, 2759, 2760, 2761, 2762, 2763, 2764, 2765, 2766, 2767, 2768, 2769, 2770, 2771, 2772, 2773, 2774, 2775, 2776, 2777, 2778, 2779, 2780, 2781, 2782, 2783, 2784, 2785, 2786, 2787, 2788, 2789, 2790, 2791, 2792, 2793, 2794, 2795, 2796, 2797, 2798, 2799, 2800, 2801, 2802, 2803, 2804, 2805, 2806, 2807, 2808, 2809, 2810, 2811, 2812, 2813, 2814, 2815, 2816, 2817, 2818, 2819, 2820, 2821, 2822, 2823, 2824, 2825, 2826, 2827, 2828, 2829, 2830, 2831, 2832, 2833, 2834, 2835, 2836, 2837, 2838, 2839, 2840, 2841, 2842, 2843, 2844, 2845, 2846, 2847, 2848, 2849, 2850, 2851, 2852, 2853, 2854, 2855, 2856, 2857, 2858, 2859, 2860, 2861, 2862, 2863, 2864, 2865, 2866, 2867, 2868, 2869, 2870, 2871, 2872, 2873, 2874, 2875, 2876, 2877, 2878, 2879, 2880, 2881, 2882, 2883, 2884, 2885, 2886, 2887, 2888, 2889, 2890, 2891, 2892, 2893, 2894, 2895, 2896, 2897, 2898, 2899, 2900, 2901, 2902, 2903, 2904, 2905, 2906, 2907, 2908, 2909, 2910, 2911, 2912, 2913, 2914, 2915, 2916, 2917, 2918, 2919, 2920, 2921, 2922, 2923, 2924, 2925, 2926, 2927, 2928, 2929, 2930, 2932, 2933, 2934, 2935, 2936, 2937, 2938, 2939, 2940, 2941, 2942, 2943, 2944, 2945, 2946, 2947, 2948, 2949, 2950, 2951, 2952, 2953, 2954, 2955, 2956, 2957, 2958, 2959, 2960, 2961, 2962, 2963, 2964, 2965, 2966, 2967, 2968, 2969, 2970, 2971, 2972, 2973, 2974, 2975, 2976, 2977, 2978, 2979, 2980, 2981, 2982, 2983, 2984, 2985, 2986, 2987, 2988, 2989, 2990, 2991, 2992, 2993, 2994, 2995, 2996, 2997, 2998, 2999, 3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 3012, 3013, 3014, 3015, 3016, 3017, 3018, 3019, 3020, 3021, 3022, 3023, 3024, 3025, 3026, 3027, 3028, 3029, 3030, 3031, 3032, 3033, 3034, 3035, 3036, 3037, 3038, 3039, 3040, 3041, 3042, 3043, 3044, 3045, 3046, 3047, 3048, 3049, 3050, 3051, 3052, 3053, 3054, 3055, 3056, 3057, 3058, 3059, 3060, 3061, 3062, 3063, 3064, 3065, 3066, 3067, 3068, 3069, 3070, 3071, 3072, 3073, 3074, 3075, 3076, 3077, 3078, 3079, 3080, 3081, 3082, 3083, 3084, 3085, 3086, 3087, 3088, 3089, 3090, 3091, 3092, 3093, 3094, 3095, 3096, 3097, 3098, 3099, 3100, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110, 3111, 3112, 3113, 3114, 3115, 3116, 3117, 3118, 3119, 3120, 3121, 3122, 3123, 3124, 3125, 3126, 3127, 3128, 3129, 3130, 3131, 3132, 3133, 3134, 3135, 3136, 3137, 3138, 3139, 3140, 3141, 3142, 3143, 3144, 3145, 3146, 3147, 3148, 3149, 3150, 3151, 3152, 3153, 3154, 3155, 3156, 3157, 3158, 3159, 3160, 3161, 3162, 3163, 3164, 3165, 3166, 3167, 3168, 3169, 3170, 3171, 3172, 3173, 3174, 3175, 3176, 3177, 3178, 3179, 3180, 3181, 3182, 3183, 3184, 3185, 3186, 3187, 3188, 3189, 3190, 3191, 3192, 3193, 3194, 3195, 3196, 3197, 3198, 3199, 3200, 3201, 3202, 3203, 3204, 3205, 3206, 3207, 3208, 3209, 3210, 3211, 3212, 3213, 3214, 3215, 3216, 3217, 3218, 3219, 3220, 3221, 3222, 3223, 3224, 3225, 3226, 3227, 3228, 3229, 3230, 3231, 3232, 3233, 3234, 3235, 3236, 3237, 3238, 3239, 3240, 3241, 3242, 3243, 3244, 3245, 3246, 3247, 3248, 3249, 3250, 3251, 3252, 3253, 3254, 3255, 3256, 3257, 3258, 3259, 3260, 3261, 3262, 3263, 3264, 3265, 3266, 3267, 3268, 3269, 3270, 3271, 3272, 3273, 3274, 3275, 3276, 3277, 3278, 3279, 3280, 3281, 3282, 3283, 3284, 3285, 3286, 3287, 3288, 3289, 3290, 3291, 3292, 3293, 3294, 3295, 3296, 3297, 3298, 3299, 3300, 3301, 3302, 3303, 3304, 3305, 3306, 3307, 3308, 3309, 3310, 3311, 3312, 3313, 3314, 3315, 3316, 3317, 3318, 3319, 3320, 3321, 3322, 3323, 3324, 3325, 3326, 3327, 3328, 3329, 3330, 3331, 3332, 3333, 3334, 3335, 3336, 3337, 3338, 3339, 3340, 3341, 3342, 3343, 3344, 3345, 3346, 3347, 3348, 3349, 3350, 3351, 3352, 3353, 3354, 3355, 3356, 3357, 3358, 3359, 3360, 3361, 3362, 3363, 3364, 3365, 3366, 3367, 3368, 3369, 3370, 3371, 3372, 3373, 3374, 3375, 3376, 3377, 3378, 3379, 3380, 3381, 3382, 3383, 3384, 3385, 3386, 3387, 3388, 3389, 3390, 3391, 3392, 3393, 3394, 3395, 3396, 3397, 3398, 3399, 3400, 3401, 3402, 3403, 3404, 3405, 3406, 3407, 3408, 3409, 3410, 3411, 3412, 3413, 3414, 3415, 3416, 3417, 3418, 3419, 3420, 3421, 3422, 3423, 3424, 3425, 3426, 3427, 3428, 3429, 3430, 3431, 3432, 3433, 3434, 3435, 3436, 3437, 3438, 3439, 3440, 3441, 3442, 3443, 3444, 3445, 3446, 3447, 3448, 3449, 3450, 3451, 3452, 3453, 3454, 3455, 3456, 3457, 3458, 3459, 3460, 3461, 3462, 3463, 3464, 3465, 3466, 3467, 3468, 3469, 3470, 3471, 3472, 3473, 3474, 3475, 3476, 3477, 3478, 3479, 3480, 3481, 3482, 3483, 3484, 3485, 3486, 3487, 3488, 3489, 3490, 3491, 3492, 3493, 3494, 3495, 3496, 3497, 3498, 3499, 3500, 3501, 3502, 3503, 3504, 3505, 3506, 3507, 3508, 3509, 3510, 3511, 3512, 3513, 3514, 3515, 3516, 3517, 3518, 3519, 3520, 3521, 3522, 3523, 3524, 3525, 3526, 3527, 3528, 3529, 3530, 3531, 3532, 3533, 3534, 3535, 3536, 3537, 3538, 3539, 3540, 3541, 3542, 3543, 3544, 3545, 3546, 3547, 3548, 3549, 3550, 3551, 3552, 3553, 3554, 3555, 3556, 3557, 3558, 3559, 3560, 3561, 3562, 3563, 3564, 3565, 3566, 3567, 3568, 3569, 3570, 3571, 3572, 3573, 3574, 3575, 3576, 3577, 3578, 3579, 3580, 3581, 3582, 3583, 3584, 3585, 3586, 3587, 3588, 3589, 3590, 3591, 3592, 3593, 3594, 3595, 3596, 3597, 3598, 3599, 3600, 3601, 3602, 3603, 3604, 3605, 3606, 3607, 3608, 3609, 3610, 3611, 3612, 3613, 3614, 3615, 3616, 3617, 3618, 3619, 3620, 3621, 3622, 3623, 3624, 3625, 3626, 3627, 3628, 3629, 3630, 3631, 3632, 3633, 3634, 3635, 3636, 3637, 3638, 3639, 3640, 3641, 3642, 3643, 3644, 3645, 3646, 3647, 3648, 3649, 3650, 3651, 3652, 3653, 3654, 3655, 3656, 3657, 3658, 3659, 3660, 3661, 3662, 3663, 3664, 3665, 3666, 3667, 3668, 3669, 3670, 3671, 3672, 3673, 3674, 3675, 3676, 3677, 3678, 3679, 3680, 3681, 3682, 3683, 3684, 3685, 3686, 3687, 3688, 3689, 3690, 3691, 3692, 3693, 3694, 3695, 3696, 3697, 3698, 3699, 3700, 3701, 3702, 3703, 3704, 3705, 3706, 3707, 3708, 3709, 3710, 3711, 3712, 3713, 3714, 3715, 3716, 3717, 3718, 3719, 3720, 3721, 3722, 3723, 3724, 3725, 3726, 3727, 3728, 3729, 3730, 3731, 3732, 3733, 3734, 3735, 3736, 3737, 3738, 3739, 3740, 3741, 3742, 3743, 3744, 3745, 3746, 3747, 3748, 3749, 3750, 3751, 3752, 3753, 3754, 3755, 3756, 3757, 3758, 3759, 3760, 3761, 3762, 3763, 3764, 3765, 3766, 3767, 3768, 3769, 3770, 3771, 3772, 3773, 3774, 3775, 3776, 3777, 3778, 3779, 3780, 3781, 3782, 3783, 3784, 3785, 3786, 3787, 3788, 3789, 3790, 3791, 3792, 3793, 3794, 3795, 3796, 3797, 3798, 3799, 3800, 3801, 3802, 3803, 3804, 3805, 3806, 3807, 3808, 3809, 3810, 3811, 3812, 3813, 3814, 3815, 3816, 3817, 3818, 3819, 3820, 3821, 3822, 3823, 3824, 3825, 3826, 3827, 3828, 3829, 3830, 3831, 3832, 3833, 3834, 3835, 3836, 3837, 3838, 3839, 3840, 3841, 3842, 3843, 3844, 3845, 3846, 3847, 3848, 3849, 3850, 3851, 3852, 3853, 3854, 3855, 3856, 3857, 3858, 3859, 3860, 3861, 3862, 3863, 3864, 3865, 3866, 3867, 3868, 3869, 3870, 3871, 3872, 3873, 3874, 3875, 3876, 3877, 3878, 3879, 3880, 3881, 3882, 3883, 3884, 3885, 3886, 3887, 3888, 3889, 3890, 3891, 3892, 3893, 3894, 3895, 3896, 3897, 3898, 3899, 3900, 3901, 3902, 3903, 3904, 3905, 3906, 3907, 3908, 3909, 3910, 3911, 3912, 3913, 3914, 3915, 3916, 3917, 3918, 3919, 3920, 3921, 3922, 3923, 3924, 3925, 3926, 3927, 3928, 3929, 3930, 3931, 3932, 3933, 3934, 3935, 3936, 3937, 3938, 3939, 3940, 3941, 3942, 3943, 3944, 3945, 3946, 3947, 3948, 3949, 3950, 3951, 3952, 3953, 3954, 3955, 3956, 3957, 3958, 3959, 3960, 3961, 3962, 3963, 3964, 3965, 3966, 3967, 3968, 3969, 3970, 3971, 3972, 3973, 3974, 3975, 3976, 3977, 3978, 3979, 3980, 3981, 3982, 3983, 3984, 3985, 3986, 3987, 3988, 3989, 3990, 3991, 3992, 3993, 3994, 3995, 3996, 3997, 3998, 3999, 4000, 4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008, 4009, 4010, 4011, 4012, 4013, 4014, 4015, 4016, 4017, 4018, 4019, 4020, 4021, 4022, 4023, 4024, 4025, 4026, 4027, 4028, 4029, 4030, 4031, 4032, 4033, 4034, 4035, 4036, 4037, 4038, 4039, 4040, 4041, 4042, 4043, 4044, 4045, 4046, 4047, 4048, 4049, 4050, 4051, 4052, 4053, 4054, 4055, 4056, 4057, 4058, 4059, 4060, 4061, 4062, 4063, 4064, 4065, 4066, 4067, 4068, 4069, 4070, 4071, 4072, 4073, 4074, 4075, 4076, 4077, 4078, 4079, 4080, 4081, 4082, 4083, 4084, 4085, 4086, 4087, 4088, 4089, 4090, 4091, 4092, 4093, 4094, 4095, 4096, 4097, 4098, 4099, 4100, 4101, 4102, 4103, 4104, 4105, 4106, 4107, 4108, 4109, 4110, 4111, 4112, 4113, 4114, 4115, 4116, 4117, 4118, 4119, 4120, 4121, 4122, 4123, 4124, 4125, 4126, 4127, 4128, 4129, 4130, 4131, 4132, 4133, 4134, 4135, 4136, 4137, 4138, 4139, 4140, 4141, 4142, 4143, 4144, 4145, 4146, 4147, 4148, 4149, 4150, 4151, 4152, 4153, 4154, 4155, 4156, 4157, 4158, 4159, 4160, 4161, 4162, 4163, 4164, 4165, 4166, 4167, 4168, 4169, 4170, 4171, 4172, 4173, 4174, 4175, 4176, 4177, 4178, 4179, 4180, 4181, 4182, 4183, 4184, 4185, 4186, 4187, 4188, 4189, 4190, 4191, 4192, 4193, 4194, 4195, 4196, 4197, 4198, 4199, 4200, 4201, 4202, 4203, 4204, 4205, 4206, 4207, 4208, 4209, 4210, 4211, 4212, 4213, 4214, 4215, 4216, 4217, 4218, 4219, 4220, 4221, 4222, 4223, 4224, 4225, 4226, 4227, 4228, 4229, 4230, 4231, 4232, 4233, 4234, 4235, 4236, 4237, 4238, 4239, 4240, 4241, 4242, 4243, 4244, 4245, 4246, 4247, 4248, 4249, 4250, 4251, 4252, 4253, 4254, 4255, 4256, 4257, 4258, 4259, 4260, 4261, 4262, 4263, 4264, 4265, 4266, 4267, 4268, 4269, 4270, 4271, 4272, 4273, 4274, 4275, 4276, 4277, 4278, 4279, 4280, 4281, 4282, 4283, 4284, 4285, 4286, 4287, 4288, 4289, 4290, 4291, 4292, 4293, 4294, 4295, 4296, 4297, 4298, 4299, 4300, 4301, 4302, 4303, 4304, 4305, 4306, 4307, 4308, 4309, 4310, 4311, 4312, 4313, 4314, 4315, 4316, 4317, 4318, 4319, 4320, 4321, 4322, 4323, 4324, 4325, 4326, 4327, 4328, 4329, 4330, 4331, 4332, 4333, 4334, 4335, 4336, 4337, 4338, 4339, 4340, 4341, 4342, 4343, 4344, 4345, 4346, 4347, 4348, 4349, 4350, 4351, 4352, 4353, 4354, 4355, 4356, 4357, 4358], "xaxis": "x", "y": [-11.272715951727019, -11.173977372913406, -11.075238794099791, -10.877761636472565, -10.779023057658952, -10.680284478845337, -10.581545900031724, -10.482807321218111, -10.384068742404498, -10.285330163590883, -10.18659158477727, -10.087853005963657, -9.989114427150042, -9.89037584833643, -9.791637269522816, -9.692898690709203, -9.594160111895588, -9.495421533081975, -9.396682954268362, -9.297944375454748, -9.199205796641134, -9.100467217827521, -9.001728639013908, -8.902990060200294, -8.80425148138668, -8.705512902573068, -8.606774323759453, -8.50803574494584, -8.409297166132227, -8.310558587318614, -8.211820008504999, -8.014342850877773, -7.915604272064159, -7.816865693250545, -7.718127114436932, -7.619388535623318, -7.520649956809704, -7.421911377996091, -7.323172799182478, -7.224434220368864, -7.12569564155525, -7.026957062741637, -6.928218483928023, -6.82947990511441, -6.730741326300796, -6.632002747487183, -6.533264168673569, -6.434525589859955, -6.335787011046342, -6.138309853419115, -6.0395712746055015, -5.940832695791888, -5.8420941169782745, -5.743355538164661, -5.644616959351048, -5.545878380537434, -5.447139801723821, -5.348401222910207, -5.052185486469367, -4.953446907655753, -4.854708328842139, -4.755969750028526, -4.657231171214912, -4.558492592401299, -4.459754013587685, -4.361015434774072, -4.262276855960458, -4.163538277146845, -4.064799698333231, -3.76858396189239, -3.669845383078777, -3.5711068042651632, -3.47236822545155, -3.3736296466379363, -3.2748910678243224, -3.1761524890107093, -3.0774139101970963, -2.9786753313834815, -2.8799367525698685, -2.7811981737562554, -2.6824595949426424, -2.5837210161290276, -2.4849824373154146, -2.3862438585018015, -2.2875052796881867, -2.1887667008745737, -2.0900281220609607, -1.9912895432473476, -1.8925509644337328, -1.7938123856201198, -1.6950738068065068, -1.596335227992892, -1.497596649179279, -1.3001194915520529, -1.201380912738438, -1.102642333924825, -1.003903755111212, -0.9051651762975972, -0.8064265974839842, -0.7076880186703711, -0.5102108610431433, -0.41147228222953025, -0.3127337034159172, -0.2139951246023024, -0.11525654578868938, -0.016517966975076348, 0.08222061183853668, 0.1809591906521515, 0.2796977694657645, 0.37843634827937755, 0.4771749270929906, 0.5759135059066054, 0.6746520847202184, 0.7733906635338315, 0.8721292423474463, 1.0696063999746723, 1.1683449787882854, 1.2670835576019002, 1.3658221364155132, 1.4645607152291262, 1.563299294042741, 1.662037872856354, 1.760776451669967, 1.8595150304835801, 1.958253609297195, 2.056992188110808, 2.155730766924421, 2.353207924551649, 2.451946503365262, 2.550685082178875, 2.6494236609924897, 2.7481622398061027, 2.846900818619716, 2.945639397433329, 3.0443779762469436, 3.1431165550605566, 3.2418551338741697, 3.3405937126877845, 3.4393322915013975, 3.5380708703150106, 3.6368094491286236, 3.7355480279422384, 3.8342866067558514, 3.9330251855694645, 4.031763764383079, 4.130502343196692, 4.229240922010305, 4.327979500823918, 4.426718079637533, 4.525456658451146, 4.624195237264759, 4.722933816078374, 4.821672394891987, 4.9204109737056, 5.019149552519213, 5.117888131332826, 5.216626710146439, 5.315365288960056, 5.414103867773669, 5.512842446587282, 5.611581025400895, 5.710319604214508, 5.809058183028121, 5.907796761841734, 6.0065353406553506, 6.105273919468964, 6.401489655909803, 6.500228234723416, 6.598966813537029, 6.796443971164258, 6.895182549977871, 6.993921128791484, 7.0926597076050975, 7.1913982864187105, 7.2901368652323235, 7.38887544404594, 7.487614022859553, 7.586352601673166, 7.685091180486779, 7.783829759300392, 7.882568338114005, 7.981306916927618, 8.080045495741235, 8.178784074554848, 8.277522653368461, 8.376261232182074, 8.474999810995687, 8.5737383898093, 8.672476968622913, 8.77121554743653, 8.869954126250143, 8.968692705063756, 9.067431283877369, 9.166169862690982, 9.264908441504595, 9.363647020318208, 9.462385599131824, 9.561124177945437, 9.65986275675905, 9.758601335572664, 9.857339914386277, 9.95607849319989, 10.054817072013503, 10.15355565082712, 10.351032808454345, 10.449771387267958, 10.548509966081571, 10.647248544895184, 10.745987123708797, 10.844725702522414, 10.943464281336027, 11.04220286014964, 11.140941438963253, 11.239680017776866, 11.33841859659048, 11.437157175404092, 11.535895754217705, 11.634634333031322, 11.733372911844935, 11.832111490658548, 11.93085006947216, 12.029588648285774, 12.128327227099387, 12.227065805913, 12.325804384726617, 12.42454296354023, 12.523281542353843, 12.622020121167456, 12.720758699981069, 12.819497278794682, 12.918235857608295, 13.016974436421911, 13.115713015235524, 13.214451594049137, 13.31319017286275, 13.510667330489976, 13.60940590930359, 13.708144488117206, 13.80688306693082, 13.905621645744432, 14.004360224558045, 14.103098803371658, 14.201837382185271, 14.300575960998884, 14.3993145398125, 14.498053118626114, 14.596791697439727, 14.794268855066953, 14.893007433880566, 14.991746012694179, 15.090484591507796, 15.189223170321409, 15.287961749135022, 15.386700327948635, 15.485438906762248, 15.58417748557586, 15.682916064389474, 15.78165464320309, 15.880393222016703, 15.979131800830316, 16.07787037964393, 16.176608958457543, 16.275347537271156, 16.37408611608477, 16.472824694898385, 16.571563273711998, 16.67030185252561, 16.769040431339224, 16.867779010152837, 17.065256167780063, 17.163994746593676, 17.262733325407293, 17.361471904220906, 17.46021048303452, 17.558949061848132, 17.657687640661745, 17.756426219475358, 17.85516479828897, 17.953903377102588, 18.0526419559162, 18.151380534729814, 18.250119113543427, 18.34885769235704, 18.447596271170653, 18.546334849984266, 18.645073428797883, 18.743812007611496, 18.84255058642511, 18.94128916523872, 19.040027744052335, 19.23750490167956, 19.43498205930679, 19.632459216934016, 19.73119779574763, 19.829936374561242, 19.928674953374856, 20.126152111002085, 20.224890689815698, 20.422367847442924, 20.521106426256537, 20.61984500507015, 20.718583583883767, 20.817322162697376, 20.916060741510993, 21.11353789913822, 21.212276477951836, 21.311015056765445, 21.40975363557906, 21.50849221439267, 21.607230793206288, 21.705969372019897, 21.804707950833514, 21.90344652964713, 22.00218510846074, 22.199662266087966, 22.298400844901582, 22.49587800252881, 22.594616581342425, 22.693355160156035, 22.79209373896965, 22.89083231778326, 22.989570896596877, 23.088309475410487, 23.187048054224103, 23.28578663303772, 23.38452521185133, 23.483263790664946, 23.582002369478555, 23.680740948292172, 23.878218105919398, 23.976956684733015, 24.075695263546624, 24.17443384236024, 24.27317242117385, 24.371910999987467, 24.470649578801076, 24.569388157614693, 24.66812673642831, 24.76686531524192, 24.865603894055536, 24.964342472869145, 25.16181963049637, 25.260558209309988, 25.359296788123604, 25.458035366937214, 25.55677394575083, 25.65551252456444, 25.754251103378056, 25.852989682191666, 25.951728261005282, 26.0504668398189, 26.14920541863251, 26.247943997446125, 26.346682576259735, 26.54415973388696, 26.642898312700577, 26.741636891514194, 26.840375470327803, 26.93911404914142, 27.03785262795503, 27.235329785582255, 27.334068364395872, 27.43280694320949, 27.531545522023098, 27.729022679650324, 27.82776125846394, 27.92649983727755, 28.025238416091167, 28.123976994904783, 28.222715573718393, 28.32145415253201, 28.518931310159235, 28.617669888972845, 28.71640846778646, 28.815147046600078, 28.913885625413688, 29.012624204227304, 29.111362783040914, 29.21010136185453, 29.30883994066814, 29.407578519481756, 29.506317098295373, 29.605055677108982, 29.7037942559226, 29.80253283473621, 29.901271413549825, 30.000009992363434, 30.09874857117705, 30.197487149990668, 30.296225728804277, 30.394964307617894, 30.493702886431503, 30.59244146524512, 30.69118004405873, 30.789918622872346, 30.888657201685962, 30.987395780499572, 31.184872938126798, 31.283611516940415, 31.48108867456764, 31.579827253381257, 31.678565832194867, 31.777304411008483, 31.97478156863571, 32.073520147449315, 32.17225872626294, 32.36973588389016, 32.46847446270378, 32.56721304151739, 32.665951620331, 32.76469019914461, 32.863428777958234, 33.06090593558545, 33.159644514399076, 33.258383093212686, 33.357121672026295, 33.455860250839905, 33.55459882965353, 33.65333740846714, 33.75207598728075, 33.85081456609437, 33.94955314490798, 34.04829172372159, 34.24576888134882, 34.34450746016243, 34.44324603897604, 34.838200354230494, 34.93693893304412, 35.03567751185773, 35.13441609067134, 35.23315466948496, 35.33189324829857, 35.43063182711218, 35.52937040592579, 35.62810898473941, 35.72684756355302, 35.82558614236663, 35.924324721180255, 36.023063299993865, 36.121801878807474, 36.220540457621084, 36.31927903643471, 36.41801761524832, 36.51675619406193, 36.61549477287555, 36.71423335168916, 36.81297193050277, 36.91171050931638, 37.10918766694361, 37.20792624575722, 37.306664824570845, 37.405403403384454, 37.504141982198064, 37.60288056101167, 37.7016191398253, 37.80035771863891, 37.899096297452516, 37.99783487626614, 38.09657345507975, 38.19531203389336, 38.29405061270697, 38.39278919152059, 38.4915277703342, 38.59026634914781, 38.689004927961435, 38.787743506775044, 38.88648208558865, 39.08395924321589, 39.182697822029496, 39.281436400843106, 39.38017497965673, 39.47891355847034, 39.57765213728395, 39.67639071609756, 39.77512929491118, 39.87386787372479, 39.9726064525384, 40.071345031352024, 40.170083610165634, 40.26882218897924, 40.36756076779285, 40.466299346606476, 40.565037925420086, 40.663776504233695, 40.76251508304732, 40.86125366186093, 40.95999224067454, 41.05873081948815, 41.15746939830177, 41.25620797711538, 41.35494655592899, 41.55242371355622, 41.65116229236983, 41.74990087118344, 41.848639449997066, 41.947378028810675, 42.046116607624285, 42.14485518643791, 42.24359376525152, 42.34233234406513, 42.44107092287874, 42.53980950169236, 42.63854808050597, 42.73728665931958, 42.8360252381332, 42.93476381694681, 43.03350239576042, 43.13224097457403, 43.230979553387655, 43.329718132201265, 43.428456711014874, 43.5271952898285, 43.62593386864211, 43.72467244745572, 43.823411026269326, 43.92214960508295, 44.02088818389656, 44.11962676271017, 44.21836534152379, 44.3171039203374, 44.41584249915101, 44.51458107796462, 44.613319656778245, 44.712058235591854, 44.810796814405464, 44.90953539321909, 45.0082739720327, 45.10701255084631, 45.205751129659916, 45.30448970847354, 45.40322828728715, 45.50196686610076, 45.60070544491437, 45.69944402372799, 45.7981826025416, 45.89692118135521, 45.995659760168834, 46.094398338982444, 46.19313691779605, 46.29187549660966, 46.39061407542329, 46.489352654236896, 46.588091233050505, 46.78556839067774, 46.88430696949135, 46.98304554830496, 47.08178412711858, 47.18052270593219, 47.2792612847458, 47.377999863559424, 47.47673844237303, 47.57547702118664, 47.67421560000025, 47.772954178813876, 47.871692757627486, 47.970431336441095, 48.06916991525472, 48.16790849406833, 48.26664707288194, 48.36538565169555, 48.46412423050917, 48.56286280932278, 48.66160138813639, 48.76033996695001, 48.85907854576362, 48.95781712457723, 49.05655570339084, 49.155294282204466, 49.254032861018075, 49.352771439831685, 49.45151001864531, 49.55024859745892, 49.64898717627253, 49.74772575508614, 49.84646433389976, 49.94520291271337, 50.04394149152698, 50.1426800703406, 50.24141864915421, 50.34015722796782, 50.43889580678143, 50.537634385595055, 50.636372964408665, 50.735111543222274, 50.8338501220359, 51.03132727966312, 51.130065858476726, 51.22880443729035, 51.32754301610396, 51.42628159491757, 51.52502017373119, 51.6237587525448, 51.72249733135841, 51.82123591017202, 51.919974488985645, 52.018713067799254, 52.117451646612864, 52.21619022542649, 52.3149288042401, 52.512405961867316, 52.61114454068094, 52.70988311949455, 52.80862169830816, 52.90736027712177, 53.00609885593539, 53.104837434749, 53.20357601356261, 53.30231459237622, 53.401053171189844, 53.49979175000345, 53.59853032881706, 53.697268907630686, 53.796007486444296, 53.993484644071515, 54.09222322288514, 54.28970038051236, 54.38843895932598, 54.48717753813959, 54.5859161169532, 54.68465469576681, 54.78339327458043, 54.88213185339404, 54.98087043220765, 55.079609011021276, 55.178347589834885, 55.277086168648495, 55.375824747462104, 55.57330190508934, 55.67204048390295, 55.77077906271657, 55.86951764153018, 55.96825622034379, 56.0669947991574, 56.16573337797102, 56.26447195678463, 56.36321053559824, 56.461949114411865, 56.560687693225475, 56.659426272039084, 56.758164850852694, 56.85690342966632, 56.95564200847993, 57.05438058729354, 57.15311916610716, 57.35059632373438, 57.44933490254799, 57.54807348136161, 57.64681206017522, 57.74555063898883, 57.844289217802455, 57.943027796616065, 58.041766375429674, 58.14050495424328, 58.23924353305691, 58.33798211187052, 58.436720690684126, 58.53545926949775, 58.63419784831136, 58.73293642712497, 58.83167500593858, 58.9304135847522, 59.02915216356581, 59.12789074237942, 59.226629321193045, 59.325367900006654, 59.42410647882026, 59.52284505763387, 59.6215836364475, 59.720322215261106, 59.819060794074716, 59.91779937288834, 60.01653795170195, 60.11527653051556, 60.21401510932917, 60.31275368814279, 60.4114922669564, 60.51023084577001, 60.608969424583634, 60.707708003397244, 60.80644658221085, 60.90518516102446, 61.003923739838086, 61.102662318651696, 61.201400897465305, 61.30013947627893, 61.39887805509254, 61.49761663390615, 61.59635521271976, 61.69509379153338, 61.79383237034699, 61.8925709491606, 61.991309527974224, 62.09004810678783, 62.18878668560144, 62.28752526441505, 62.386263843228676, 62.583741000855895, 62.68247957966952, 62.78121815848313, 62.87995673729674, 62.97869531611035, 63.07743389492397, 63.17617247373758, 63.27491105255119, 63.37364963136481, 63.47238821017842, 63.57112678899203, 63.66986536780564, 63.768603946619265, 63.966081104246484, 64.06481968306011, 64.16355826187372, 64.26229684068733, 64.36103541950094, 64.45977399831456, 64.55851257712817, 64.65725115594178, 64.7559897347554, 64.85472831356901, 64.95346689238262, 65.05220547119623, 65.15094405000985, 65.24968262882346, 65.34842120763707, 65.4471597864507, 65.5458983652643, 65.64463694407792, 65.74337552289153, 65.84211410170515, 65.94085268051876, 66.03959125933237, 66.13832983814599, 66.2370684169596, 66.33580699577321, 66.43454557458682, 66.53328415340044, 66.63202273221405, 66.73076131102766, 66.82949988984129, 66.9282384686549, 67.0269770474685, 67.12571562628212, 67.22445420509574, 67.32319278390935, 67.42193136272296, 67.52066994153658, 67.61940852035019, 67.7181470991638, 67.81688567797741, 67.91562425679103, 68.01436283560464, 68.11310141441825, 68.21183999323188, 68.31057857204549, 68.4093171508591, 68.5080557296727, 68.60679430848633, 68.70553288729994, 68.80427146611355, 68.90301004492717, 69.00174862374078, 69.10048720255439, 69.199225781368, 69.29796436018162, 69.39670293899523, 69.49544151780884, 69.59418009662247, 69.69291867543608, 69.79165725424969, 69.8903958330633, 69.98913441187692, 70.08787299069053, 70.18661156950414, 70.28535014831776, 70.38408872713137, 70.48282730594498, 70.58156588475859, 70.68030446357221, 70.77904304238582, 70.87778162119943, 70.97652020001306, 71.07525877882667, 71.17399735764027, 71.27273593645388, 71.37147451526751, 71.47021309408112, 71.56895167289473, 71.66769025170835, 71.76642883052196, 71.86516740933557, 71.96390598814918, 72.0626445669628, 72.16138314577641, 72.26012172459002, 72.35886030340365, 72.45759888221725, 72.55633746103086, 72.65507603984447, 72.7538146186581, 72.8525531974717, 72.95129177628532, 73.05003035509894, 73.14876893391255, 73.24750751272616, 73.34624609153977, 73.44498467035339, 73.64246182798061, 73.74120040679423, 73.83993898560784, 74.03741614323506, 74.13615472204869, 74.2348933008623, 74.3336318796759, 74.43237045848953, 74.53110903730314, 74.62984761611675, 74.72858619493036, 74.82732477374398, 74.92606335255759, 75.0248019313712, 75.12354051018482, 75.22227908899843, 75.32101766781204, 75.51849482543928, 75.61723340425289, 75.7159719830665, 75.81471056188012, 75.91344914069373, 76.01218771950734, 76.11092629832095, 76.20966487713457, 76.30840345594818, 76.40714203476179, 76.50588061357541, 76.60461919238902, 76.80209635001624, 76.90083492882987, 76.99957350764348, 77.09831208645709, 77.19705066527071, 77.29578924408432, 77.39452782289793, 77.49326640171154, 77.69074355933877, 77.78948213815238, 77.88822071696599, 77.98695929577961, 78.08569787459322, 78.28317503222046, 78.48065218984767, 78.57939076866128, 78.67812934747491, 78.77686792628852, 78.87560650510213, 78.97434508391575, 79.07308366272936, 79.17182224154297, 79.27056082035658, 79.3692993991702, 79.46803797798381, 79.56677655679742, 79.86299229323826, 79.96173087205187, 80.0604694508655, 80.1592080296791, 80.25794660849272, 80.35668518730634, 80.45542376611995, 80.55416234493356, 80.65290092374717, 80.75163950256079, 80.8503780813744, 80.94911666018801, 81.04785523900163, 81.14659381781524, 81.24533239662885, 81.34407097544246, 81.44280955425609, 81.5415481330697, 81.6402867118833, 81.73902529069693, 81.83776386951054, 81.93650244832415, 82.03524102713776, 82.13397960595138, 82.23271818476499, 82.3314567635786, 82.43019534239222, 82.52893392120583, 82.62767250001944, 82.72641107883305, 82.82514965764668, 82.92388823646029, 83.0226268152739, 83.12136539408752, 83.22010397290113, 83.31884255171474, 83.41758113052835, 83.51631970934197, 83.61505828815558, 83.71379686696919, 83.81253544578281, 83.91127402459642, 84.01001260341003, 84.10875118222364, 84.20748976103727, 84.30622833985088, 84.40496691866448, 84.50370549747811, 84.70118265510533, 84.79992123391894, 84.89865981273256, 84.99739839154617, 85.09613697035978, 85.29361412798701, 85.39235270680062, 85.58982986442786, 85.68856844324146, 85.78730702205507, 85.8860456008687, 85.98478417968231, 86.08352275849592, 86.18226133730953, 86.28099991612315, 86.37973849493676, 86.47847707375037, 86.57721565256399, 86.6759542313776, 86.77469281019121, 86.87343138900482, 86.97216996781845, 87.07090854663205, 87.16964712544566, 87.26838570425929, 87.3671242830729, 87.4658628618865, 87.56460144070012, 87.66334001951374, 87.76207859832735, 87.86081717714096, 87.95955575595458, 88.05829433476819, 88.1570329135818, 88.25577149239541, 88.35451007120903, 88.45324865002264, 88.55198722883625, 88.74946438646349, 88.8482029652771, 88.9469415440907, 89.04568012290433, 89.14441870171794, 89.24315728053155, 89.34189585934517, 89.44063443815878, 89.53937301697239, 89.638111595786, 89.73685017459962, 89.83558875341323, 89.93432733222684, 90.03306591104047, 90.13180448985408, 90.23054306866769, 90.3292816474813, 90.42802022629492, 90.52675880510853, 90.62549738392214, 90.72423596273576, 90.82297454154937, 90.92171312036298, 91.02045169917659, 91.11919027799021, 91.21792885680382, 91.31666743561743, 91.41540601443106, 91.51414459324467, 91.61288317205828, 91.71162175087188, 91.81036032968551, 91.90909890849912, 92.00783748731273, 92.10657606612635, 92.20531464493996, 92.30405322375357, 92.40279180256718, 92.5015303813808, 92.60026896019441, 92.69900753900802, 92.79774611782165, 92.89648469663526, 92.99522327544886, 93.09396185426247, 93.1927004330761, 93.29143901188971, 93.39017759070332, 93.68639332714416, 93.78513190595777, 93.88387048477139, 93.982609063585, 94.08134764239861, 94.18008622121224, 94.27882480002584, 94.37756337883945, 94.47630195765306, 94.57504053646669, 94.6737791152803, 94.7725176940939, 94.87125627290753, 94.96999485172114, 95.06873343053475, 95.16747200934836, 95.26621058816198, 95.36494916697559, 95.56242632460282, 95.75990348223004, 95.85864206104365, 95.95738063985728, 96.05611921867089, 96.1548577974845, 96.45107353392534, 96.54981211273895, 96.64855069155257, 96.74728927036618, 96.84602784917979, 96.94476642799341, 97.04350500680702, 97.14224358562063, 97.24098216443424, 97.33972074324787, 97.43845932206148, 97.53719790087509, 97.63593647968871, 97.73467505850232, 97.83341363731593, 97.93215221612954, 98.03089079494316, 98.12962937375677, 98.22836795257038, 98.327106531384, 98.52458368901122, 98.62332226782483, 98.72206084663846, 98.82079942545207, 98.91953800426568, 99.11701516189291, 99.21575374070652, 99.31449231952013, 99.41323089833375, 99.51196947714736, 99.61070805596097, 99.7094466347746, 99.8081852135882, 99.90692379240181, 100.00566237121542, 100.10440095002905, 100.20313952884266, 100.30187810765626, 100.40061668646989, 100.4993552652835, 100.59809384409711, 100.69683242291072, 100.79557100172434, 100.89430958053795, 100.99304815935156, 101.09178673816518, 101.19052531697879, 101.2892638957924, 101.38800247460601, 101.48674105341964, 101.58547963223324, 101.68421821104685, 101.78295678986048, 101.88169536867409, 102.0791725263013, 102.17791110511493, 102.27664968392854, 102.37538826274215, 102.47412684155576, 102.57286542036938, 102.67160399918299, 102.7703425779966, 102.86908115681022, 102.96781973562383, 103.06655831443744, 103.16529689325105, 103.26403547206468, 103.36277405087829, 103.4615126296919, 103.56025120850552, 103.65898978731913, 103.75772836613274, 103.85646694494635, 103.95520552375997, 104.05394410257358, 104.15268268138719, 104.25142126020081, 104.35015983901442, 104.44889841782803, 104.54763699664164, 104.64637557545527, 104.74511415426888, 104.84385273308249, 104.94259131189611, 105.04132989070972, 105.14006846952333, 105.23880704833694, 105.33754562715056, 105.43628420596417, 105.53502278477778, 105.6337613635914, 105.73249994240501, 105.83123852121862, 105.92997710003223, 106.02871567884586, 106.12745425765947, 106.22619283647307, 106.3249314152867, 106.42366999410031, 106.52240857291392, 106.62114715172753, 106.71988573054115, 106.81862430935476, 106.91736288816837, 107.016101466982, 107.1148400457956, 107.21357862460921, 107.31231720342282, 107.41105578223645, 107.50979436105006, 107.60853293986366, 107.70727151867729, 107.8060100974909, 107.90474867630451, 108.00348725511812, 108.10222583393174, 108.20096441274535, 108.29970299155896, 108.39844157037258, 108.49718014918619, 108.5959187279998, 108.69465730681341, 108.79339588562704, 108.89213446444064, 108.99087304325425, 109.08961162206788, 109.18835020088149, 109.2870887796951, 109.3858273585087, 109.48456593732233, 109.58330451613594, 109.68204309494955, 109.78078167376317, 109.87952025257678, 109.97825883139039, 110.076997410204, 110.17573598901762, 110.27447456783123, 110.37321314664484, 110.47195172545847, 110.57069030427208, 110.66942888308569, 110.7681674618993, 110.86690604071292, 110.96564461952653, 111.06438319834014, 111.16312177715376, 111.26186035596737, 111.36059893478098, 111.45933751359459, 111.55807609240821, 111.65681467122182, 111.75555325003543, 111.85429182884906, 111.95303040766267, 112.05176898647628, 112.15050756528989, 112.34798472291712, 112.44672330173073, 112.54546188054435, 112.64420045935796, 112.84167761698518, 112.9404161957988, 113.03915477461241, 113.13789335342602, 113.23663193223965, 113.33537051105326, 113.43410908986687, 113.53284766868047, 113.6315862474941, 113.73032482630771, 113.82906340512132, 113.92780198393494, 114.02654056274855, 114.12527914156216, 114.22401772037577, 114.3227562991894, 114.421494878003, 114.52023345681661, 114.61897203563024, 114.71771061444385, 114.81644919325745, 115.01392635088469, 115.1126649296983, 115.2114035085119, 115.31014208732553, 115.40888066613914, 115.50761924495275, 115.60635782376636, 115.70509640257998, 115.80383498139359, 115.9025735602072, 116.00131213902083, 116.10005071783443, 116.19878929664804, 116.29752787546165, 116.39626645427528, 116.49500503308889, 116.5937436119025, 116.69248219071612, 116.79122076952972, 116.88995934834334, 117.08743650597056, 117.18617508478418, 117.2849136635978, 117.3836522424114, 117.48239082122502, 117.58112940003865, 117.67986797885224, 117.77860655766587, 117.87734513647946, 117.97608371529309, 118.07482229410671, 118.1735608729203, 118.27229945173393, 118.56851518817477, 118.6672537669884, 118.76599234580199, 118.86473092461561, 118.96346950342924, 119.06220808224283, 119.16094666105646, 119.25968523987005, 119.35842381868368, 119.4571623974973, 119.5559009763109, 119.65463955512452, 119.85211671275174, 119.95085529156536, 120.04959387037898, 120.14833244919258, 120.2470710280062, 120.34580960681983, 120.44454818563342, 120.54328676444705, 120.64202534326064, 120.74076392207427, 120.83950250088789, 120.93824107970148, 121.03697965851511, 121.13571823732873, 121.23445681614233, 121.33319539495595, 121.43193397376957, 121.53067255258317, 121.6294111313968, 121.72814971021042, 121.82688828902401, 121.92562686783764, 122.02436544665123, 122.12310402546485, 122.22184260427848, 122.32058118309207, 122.4193197619057, 122.51805834071932, 122.61679691953292, 122.71553549834654, 122.81427407716016, 122.91301265597376, 123.01175123478738, 123.110489813601, 123.2092283924146, 123.30796697122823, 123.40670555004182, 123.50544412885544, 123.60418270766907, 123.70292128648266, 123.80165986529629, 123.90039844410991, 123.9991370229235, 124.09787560173713, 124.19661418055075, 124.29535275936435, 124.39409133817797, 124.4928299169916, 124.59156849580519, 124.78904565343241, 124.88778423224603, 124.98652281105966, 125.08526138987325, 125.18399996868688, 125.2827385475005, 125.3814771263141, 125.48021570512772, 125.57895428394134, 125.67769286275494, 125.77643144156856, 125.87517002038219, 126.0726471780094, 126.171385756823, 126.27012433563662, 126.36886291445025, 126.46760149326384, 126.56634007207747, 126.66507865089109, 126.76381722970469, 126.86255580851831, 126.96129438733193, 127.06003296614553, 127.15877154495915, 127.25751012377278, 127.35624870258637, 127.55372586021359, 127.65246443902721, 127.75120301784084, 127.84994159665443, 127.94868017546806, 128.1461573330953, 128.2448959119089, 128.34363449072254, 128.44237306953613, 128.54111164834976, 128.63985022716338, 128.73858880597697, 128.8373273847906, 128.9360659636042, 129.03480454241782, 129.13354312123144, 129.23228170004504, 129.33102027885866, 129.42975885767228, 129.52849743648588, 129.6272360152995, 129.72597459411313, 129.82471317292672, 129.92345175174034, 130.02219033055397, 130.12092890936756, 130.2196674881812, 130.31840606699478, 130.4171446458084, 130.51588322462203, 130.61462180343563, 130.71336038224925, 130.81209896106287, 130.91083753987647, 131.0095761186901, 131.10831469750372, 131.2070532763173, 131.30579185513093, 131.40453043394456, 131.50326901275815, 131.60200759157178, 131.70074617038537, 131.799484749199, 131.89822332801262, 131.99696190682621, 132.09570048563984, 132.19443906445346, 132.29317764326706, 132.39191622208068, 132.4906548008943, 132.5893933797079, 132.68813195852152, 132.78687053733515, 132.88560911614874, 132.98434769496237, 133.08308627377596, 133.18182485258959, 133.2805634314032, 133.3793020102168, 133.47804058903043, 133.57677916784405, 133.67551774665765, 133.77425632547127, 133.8729949042849, 133.9717334830985, 134.0704720619121, 134.16921064072574, 134.26794921953933, 134.36668779835296, 134.46542637716655, 134.56416495598017, 134.6629035347938, 134.7616421136074, 134.86038069242102, 135.05785785004824, 135.15659642886186, 135.25533500767548, 135.35407358648908, 135.4528121653027, 135.55155074411633, 135.65028932292992, 135.84776648055714, 135.94650505937076, 136.0452436381844, 136.14398221699798, 136.2427207958116, 136.34145937462523, 136.44019795343883, 136.53893653225245, 136.63767511106607, 136.73641368987967, 136.8351522686933, 136.93389084750692, 137.0326294263205, 137.13136800513414, 137.23010658394773, 137.32884516276135, 137.42758374157498, 137.52632232038857, 137.6250608992022, 137.72379947801582, 137.82253805682942, 137.92127663564304, 138.02001521445666, 138.11875379327026, 138.21749237208388, 138.3162309508975, 138.4149695297111, 138.51370810852472, 138.61244668733832, 138.71118526615194, 138.80992384496557, 138.90866242377916, 139.0074010025928, 139.1061395814064, 139.20487816022, 139.30361673903363, 139.40235531784725, 139.50109389666085, 139.59983247547447, 139.6985710542881, 139.7973096331017, 139.8960482119153, 139.9947867907289, 140.09352536954253, 140.19226394835616, 140.29100252716975, 140.38974110598338, 140.488479684797, 140.5872182636106, 140.68595684242422, 140.78469542123784, 140.88343400005144, 140.98217257886506, 141.08091115767866, 141.17964973649228, 141.2783883153059, 141.3771268941195, 141.47586547293312, 141.57460405174675, 141.67334263056034, 141.77208120937397, 141.8708197881876, 141.96955836700118, 142.0682969458148, 142.16703552462843, 142.26577410344203, 142.36451268225565, 142.46325126106925, 142.56198983988287, 142.7594669975101, 142.8582055763237, 142.95694415513734, 143.05568273395093, 143.15442131276455, 143.25315989157818, 143.35189847039177, 143.4506370492054, 143.54937562801902, 143.64811420683262, 143.74685278564624, 143.84559136445984, 143.94432994327346, 144.04306852208708, 144.14180710090068, 144.2405456797143, 144.33928425852793, 144.43802283734152, 144.53676141615514, 144.63549999496877, 144.73423857378236, 144.832977152596, 144.9317157314096, 145.0304543102232, 145.12919288903683, 145.22793146785042, 145.32667004666405, 145.42540862547767, 145.52414720429127, 145.6228857831049, 145.8203629407321, 145.91910151954573, 146.01784009835936, 146.11657867717295, 146.21531725598658, 146.3140558348002, 146.4127944136138, 146.70901015005464, 146.80774872886826, 146.90648730768186, 147.00522588649548, 147.2027030441227, 147.30144162293632, 147.40018020174995, 147.49891878056354, 147.59765735937717, 147.6963959381908, 147.79513451700439, 147.893873095818, 147.9926116746316, 148.09135025344523, 148.19008883225885, 148.28882741107245, 148.38756598988607, 148.4863045686997, 148.5850431475133, 148.6837817263269, 148.88125888395413, 148.97999746276776, 149.07873604158138, 149.17747462039497, 149.2762131992086, 149.3749517780222, 149.47369035683582, 149.57242893564944, 149.67116751446304, 149.76990609327666, 149.86864467209028, 149.96738325090388, 150.0661218297175, 150.16486040853113, 150.26359898734472, 150.36233756615835, 150.46107614497197, 150.55981472378556, 150.6585533025992, 150.75729188141278, 150.8560304602264, 150.95476903904003, 151.05350761785363, 151.15224619666725, 151.25098477548087, 151.34972335429447, 151.4484619331081, 151.54720051192172, 151.6459390907353, 151.74467766954893, 151.84341624836256, 151.94215482717615, 152.04089340598978, 152.13963198480337, 152.238370563617, 152.33710914243062, 152.43584772124422, 152.53458630005784, 152.63332487887146, 152.73206345768506, 152.83080203649868, 152.9295406153123, 153.0282791941259, 153.12701777293952, 153.22575635175315, 153.32449493056674, 153.42323350938037, 153.52197208819396, 153.6207106670076, 153.7194492458212, 153.8181878246348, 153.91692640344843, 154.01566498226205, 154.11440356107565, 154.21314213988927, 154.3118807187029, 154.5093578763301, 154.60809645514374, 154.70683503395733, 154.80557361277096, 154.90431219158455, 155.00305077039818, 155.1017893492118, 155.2005279280254, 155.29926650683902, 155.39800508565264, 155.49674366446624, 155.59548224327986, 155.69422082209348, 155.79295940090708, 155.8916979797207, 155.99043655853433, 156.08917513734792, 156.18791371616155, 156.28665229497514, 156.38539087378877, 156.4841294526024, 156.58286803141598, 156.6816066102296, 156.78034518904323, 156.87908376785683, 156.97782234667045, 157.07656092548407, 157.17529950429767, 157.2740380831113, 157.37277666192492, 157.4715152407385, 157.57025381955214, 157.66899239836573, 157.76773097717935, 157.86646955599298, 157.96520813480657, 158.0639467136202, 158.16268529243382, 158.26142387124742, 158.36016245006104, 158.45890102887466, 158.55763960768826, 158.65637818650188, 158.7551167653155, 158.8538553441291, 158.95259392294273, 159.05133250175632, 159.15007108056994, 159.24880965938357, 159.34754823819716, 159.4462868170108, 159.5450253958244, 159.643763974638, 159.74250255345163, 159.84124113226525, 160.03871828989247, 160.1374568687061, 160.2361954475197, 160.33493402633331, 160.4336726051469, 160.53241118396053, 160.63114976277416, 160.72988834158775, 160.82862692040138, 160.927365499215, 161.0261040780286, 161.12484265684222, 161.22358123565584, 161.32231981446944, 161.42105839328306, 161.51979697209669, 161.61853555091028, 161.7172741297239, 161.8160127085375, 161.91475128735112, 162.01348986616475, 162.11222844497834, 162.21096702379197, 162.3097056026056, 162.40844418141918, 162.5071827602328, 162.60592133904643, 162.70465991786003, 162.80339849667365, 162.90213707548727, 163.00087565430087, 163.0996142331145, 163.1983528119281, 163.2970913907417, 163.39582996955534, 163.49456854836893, 163.59330712718256, 163.69204570599618, 163.79078428480977, 163.8895228636234, 163.98826144243702, 164.08700002125062, 164.18573860006424, 164.28447717887786, 164.38321575769146, 164.48195433650508, 164.58069291531868, 164.6794314941323, 164.77817007294593, 164.87690865175952, 164.97564723057314, 165.07438580938677, 165.17312438820036, 165.271862967014, 165.3706015458276, 165.4693401246412, 165.56807870345483, 165.66681728226845, 165.76555586108205, 165.86429443989567, 165.96303301870927, 166.0617715975229, 166.16051017633652, 166.2592487551501, 166.35798733396373, 166.45672591277736, 166.55546449159095, 166.65420307040458, 166.7529416492182, 166.8516802280318, 166.95041880684542, 167.04915738565902, 167.14789596447264, 167.24663454328626, 167.34537312209986, 167.44411170091348, 167.5428502797271, 167.6415888585407, 167.74032743735432, 167.83906601616795, 167.93780459498154, 168.03654317379517, 168.1352817526088, 168.23402033142239, 168.332758910236, 168.4314974890496, 168.53023606786323, 168.62897464667685, 168.72771322549045, 168.82645180430407, 168.9251903831177, 169.0239289619313, 169.1226675407449, 169.22140611955854, 169.32014469837213, 169.41888327718576, 169.51762185599938, 169.61636043481298, 169.7150990136266, 169.8138375924402, 169.91257617125382, 170.01131475006744, 170.11005332888104, 170.20879190769466, 170.30753048650828, 170.40626906532188, 170.5050076441355, 170.60374622294913, 170.70248480176272, 170.80122338057635, 170.89996195938997, 171.0974391170172, 171.19617769583078, 171.2949162746444, 171.39365485345803, 171.49239343227163, 171.59113201108525, 171.68987058989887, 171.78860916871247, 171.8873477475261, 171.98608632633972, 172.0848249051533, 172.18356348396694, 172.28230206278056, 172.38104064159415, 172.47977922040778, 172.57851779922137, 172.677256378035, 172.77599495684862, 172.87473353566222, 172.97347211447584, 173.07221069328946, 173.17094927210306, 173.26968785091668, 173.3684264297303, 173.4671650085439, 173.56590358735752, 173.66464216617115, 173.76338074498474, 173.86211932379837, 173.96085790261196, 174.0595964814256, 174.1583350602392, 174.2570736390528, 174.35581221786643, 174.45455079668005, 174.55328937549365, 174.65202795430727, 174.7507665331209, 174.8495051119345, 174.94824369074811, 175.04698226956174, 175.14572084837533, 175.24445942718896, 175.34319800600255, 175.44193658481618, 175.5406751636298, 175.6394137424434, 175.73815232125702, 175.83689090007064, 175.93562947888424, 176.03436805769786, 176.13310663651149, 176.23184521532508, 176.3305837941387, 176.42932237295233, 176.62679953057955, 176.72553810939314, 176.82427668820677, 176.9230152670204, 177.02175384583398, 177.1204924246476, 177.21923100346123, 177.31796958227483, 177.41670816108845, 177.51544673990207, 177.61418531871567, 177.7129238975293, 177.81166247634292, 177.9104010551565, 178.00913963397014, 178.10787821278373, 178.20661679159736, 178.30535537041098, 178.40409394922457, 178.5028325280382, 178.60157110685182, 178.70030968566542, 178.79904826447904, 178.89778684329266, 178.99652542210626, 179.09526400091988, 179.1940025797335, 179.2927411585471, 179.39147973736073, 179.49021831617432, 179.58895689498794, 179.68769547380157, 179.78643405261516, 179.8851726314288, 179.9839112102424, 180.082649789056, 180.18138836786963, 180.28012694668325, 180.37886552549685, 180.47760410431047, 180.5763426831241, 180.6750812619377, 180.77381984075132, 180.8725584195649, 180.97129699837853, 181.07003557719216, 181.16877415600575, 181.26751273481938, 181.366251313633, 181.4649898924466, 181.56372847126022, 181.66246705007384, 181.76120562888744, 181.85994420770106, 181.9586827865147, 182.05742136532828, 182.1561599441419, 182.2548985229555, 182.35363710176912, 182.45237568058275, 182.55111425939634, 182.64985283820997, 182.7485914170236, 182.84732999583719, 182.9460685746508, 183.04480715346443, 183.14354573227803, 183.24228431109165, 183.34102288990528, 183.43976146871887, 183.5385000475325, 183.6372386263461, 183.7359772051597, 183.83471578397334, 183.93345436278693, 184.03219294160056, 184.13093152041418, 184.22967009922777, 184.3284086780414, 184.42714725685502, 184.52588583566862, 184.62462441448224, 184.72336299329586, 184.82210157210946, 184.92084015092308, 185.01957872973668, 185.1183173085503, 185.21705588736393, 185.31579446617752, 185.41453304499115, 185.51327162380477, 185.61201020261836, 185.710748781432, 185.8094873602456, 185.9082259390592, 186.00696451787283, 186.10570309668645, 186.20444167550005, 186.30318025431367, 186.40191883312727, 186.59939599075452, 186.6981345695681, 186.79687314838174, 186.89561172719536, 186.99435030600895, 187.09308888482258, 187.1918274636362, 187.2905660424498, 187.38930462126342, 187.48804320007704, 187.58678177889064, 187.68552035770426, 187.78425893651786, 187.88299751533148, 187.9817360941451, 188.0804746729587, 188.17921325177232, 188.27795183058595, 188.37669040939954, 188.47542898821317, 188.5741675670268, 188.6729061458404, 188.771644724654, 188.87038330346763, 188.96912188228123, 189.06786046109485, 189.16659903990845, 189.26533761872207, 189.3640761975357, 189.4628147763493, 189.5615533551629, 189.66029193397654, 189.75903051279013, 189.85776909160376, 189.95650767041738, 190.05524624923098, 190.1539848280446, 190.25272340685822, 190.35146198567182, 190.45020056448544, 190.54893914329904, 190.64767772211266, 190.74641630092628, 190.84515487973988, 190.9438934585535, 191.04263203736713, 191.14137061618072, 191.24010919499435, 191.33884777380797, 191.43758635262157, 191.5363249314352, 191.63506351024878, 191.7338020890624, 191.83254066787603, 191.93127924668963, 192.03001782550325, 192.12875640431687, 192.22749498313047, 192.3262335619441, 192.42497214075772, 192.5237107195713, 192.62244929838494, 192.72118787719856, 192.81992645601215, 192.91866503482578, 193.01740361363937, 193.116142192453, 193.21488077126662, 193.31361935008022, 193.41235792889384, 193.51109650770746, 193.60983508652106, 193.70857366533468, 193.8073122441483, 193.9060508229619, 194.00478940177553, 194.10352798058915, 194.20226655940274, 194.30100513821637, 194.39974371702996, 194.4984822958436, 194.5972208746572, 194.6959594534708, 194.79469803228443, 194.89343661109805, 194.99217518991165, 195.09091376872527, 195.1896523475389, 195.2883909263525, 195.38712950516611, 195.48586808397974, 195.58460666279333, 195.68334524160696, 195.78208382042055, 195.88082239923418, 195.9795609780478, 196.0782995568614, 196.17703813567502, 196.27577671448864, 196.37451529330224, 196.47325387211586, 196.57199245092949, 196.67073102974308, 196.7694696085567, 196.86820818737033, 196.96694676618392, 197.06568534499755, 197.16442392381114, 197.26316250262477, 197.3619010814384, 197.46063966025199, 197.5593782390656, 197.65811681787923, 197.75685539669283, 197.85559397550645, 197.95433255432008, 198.05307113313367, 198.1518097119473, 198.25054829076092, 198.3492868695745, 198.44802544838814, 198.54676402720173, 198.64550260601536, 198.74424118482898, 198.84297976364257, 198.9417183424562, 199.04045692126982, 199.13919550008342, 199.23793407889704, 199.33667265771066, 199.43541123652426, 199.53414981533788, 199.6328883941515, 199.7316269729651, 199.83036555177873, 199.92910413059232, 200.02784270940595, 200.12658128821957, 200.22531986703316, 200.3240584458468, 200.4227970246604, 200.521535603474, 200.62027418228763, 200.71901276110125, 200.81775133991485, 200.91648991872847, 201.0152284975421, 201.1139670763557, 201.21270565516932, 201.3114442339829, 201.41018281279653, 201.50892139161016, 201.60765997042375, 201.70639854923738, 201.805137128051, 201.9038757068646, 202.00261428567822, 202.10135286449184, 202.20009144330544, 202.29883002211906, 202.3975686009327, 202.49630717974628, 202.5950457585599, 202.6937843373735, 202.79252291618712, 202.89126149500075, 202.99000007381434, 203.08873865262797, 203.1874772314416, 203.2862158102552, 203.3849543890688, 203.48369296788243, 203.58243154669603, 203.68117012550965, 203.77990870432328, 203.87864728313687, 203.9773858619505, 204.0761244407641, 204.1748630195777, 204.27360159839134, 204.37234017720493, 204.47107875601856, 204.56981733483218, 204.66855591364578, 204.7672944924594, 204.86603307127302, 204.96477165008662, 205.06351022890024, 205.16224880771387, 205.26098738652746, 205.35972596534108, 205.45846454415468, 205.5572031229683, 205.65594170178193, 205.75468028059552, 205.85341885940915, 205.95215743822277, 206.05089601703637, 206.14963459585, 206.2483731746636, 206.3471117534772, 206.44585033229083, 206.54458891110446, 206.64332748991805, 206.74206606873167, 206.84080464754527, 206.9395432263589, 207.03828180517252, 207.1370203839861, 207.23575896279974, 207.33449754161336, 207.43323612042695, 207.53197469924058, 207.6307132780542, 207.7294518568678, 207.82819043568142, 207.92692901449504, 208.02566759330864, 208.12440617212226, 208.22314475093586, 208.32188332974948, 208.4206219085631, 208.5193604873767, 208.61809906619033, 208.71683764500395, 208.81557622381754, 208.91431480263117, 209.0130533814448, 209.1117919602584, 209.210530539072, 209.30926911788563, 209.40800769669923, 209.50674627551285, 209.60548485432645, 209.70422343314007, 209.8029620119537, 209.9017005907673, 210.00043916958091, 210.09917774839454, 210.19791632720813, 210.29665490602176, 210.39539348483538, 210.49413206364898, 210.5928706424626, 210.69160922127622, 210.79034780008982, 210.88908637890344, 210.98782495771704, 211.08656353653066, 211.18530211534429, 211.28404069415788, 211.3827792729715, 211.48151785178513, 211.58025643059872, 211.67899500941235, 211.77773358822597, 211.87647216703957, 211.9752107458532, 212.0739493246668, 212.1726879034804, 212.27142648229403, 212.37016506110763, 212.46890363992125, 212.56764221873487, 212.66638079754847, 212.7651193763621, 212.86385795517572, 212.9625965339893, 213.06133511280294, 213.16007369161656, 213.25881227043016, 213.35755084924378, 213.4562894280574, 213.555028006871, 213.65376658568462, 213.75250516449822, 213.85124374331184, 213.94998232212546, 214.04872090093906, 214.14745947975268, 214.2461980585663, 214.3449366373799, 214.44367521619353, 214.54241379500715, 214.64115237382074, 214.73989095263437, 214.838629531448, 214.9373681102616, 215.0361066890752, 215.1348452678888, 215.23358384670243, 215.33232242551605, 215.43106100432965, 215.52979958314327, 215.6285381619569, 215.7272767407705, 215.82601531958412, 215.92475389839774, 216.02349247721133, 216.12223105602496, 216.22096963483855, 216.31970821365218, 216.4184467924658, 216.5171853712794, 216.61592395009302, 216.71466252890664, 216.81340110772024, 216.91213968653386, 217.0108782653475, 217.10961684416108, 217.2083554229747, 217.30709400178833, 217.40583258060192, 217.50457115941555, 217.60330973822914, 217.70204831704277, 217.8007868958564, 217.89952547466999, 217.9982640534836, 218.09700263229723, 218.19574121111083, 218.29447978992445, 218.39321836873808, 218.49195694755167, 218.5906955263653, 218.68943410517892, 218.7881726839925, 218.88691126280614, 218.98564984161973, 219.08438842043336, 219.18312699924698, 219.28186557806058, 219.3806041568742, 219.47934273568782, 219.57808131450142, 219.67681989331504, 219.87429705094226, 219.97303562975588, 220.0717742085695, 220.1705127873831, 220.26925136619673, 220.36798994501032, 220.46672852382395, 220.56546710263757, 220.66420568145116, 220.7629442602648, 220.8616828390784, 220.960421417892, 221.05915999670563, 221.15789857551925, 221.25663715433285, 221.35537573314647, 221.4541143119601, 221.5528528907737, 221.65159146958732, 221.7503300484009, 221.84906862721454, 221.94780720602816, 222.04654578484175, 222.14528436365538, 222.244022942469, 222.3427615212826, 222.44150010009622, 222.54023867890984, 222.63897725772344, 222.73771583653706, 222.8364544153507, 222.93519299416428, 223.0339315729779, 223.1326701517915, 223.23140873060512, 223.33014730941875, 223.42888588823234, 223.52762446704597, 223.6263630458596, 223.7251016246732, 223.8238402034868, 223.92257878230043, 224.02131736111403, 224.12005593992765, 224.21879451874128, 224.31753309755487, 224.4162716763685, 224.5150102551821, 224.61374883399571, 224.71248741280934, 224.81122599162293, 224.90996457043656, 225.00870314925018, 225.10744172806378, 225.2061803068774, 225.30491888569102, 225.40365746450462, 225.50239604331824, 225.60113462213187, 225.69987320094546, 225.79861177975909, 225.89735035857268, 225.9960889373863, 226.09482751619993, 226.19356609501352, 226.29230467382715, 226.39104325264077, 226.48978183145437, 226.588520410268, 226.6872589890816, 226.7859975678952, 226.88473614670883, 226.98347472552246, 227.08221330433605, 227.18095188314967, 227.27969046196327, 227.3784290407769, 227.47716761959052, 227.5759061984041, 227.67464477721774, 227.77338335603136, 227.87212193484496, 227.97086051365858, 228.0695990924722, 228.1683376712858, 228.26707625009942, 228.36581482891305, 228.56329198654026, 228.66203056535386, 228.76076914416748, 228.8595077229811, 228.9582463017947, 229.05698488060833, 229.15572345942195, 229.25446203823554, 229.35320061704917, 229.4519391958628, 229.5506777746764, 229.64941635349, 229.74815493230363, 229.84689351111723, 229.94563208993085, 230.04437066874445, 230.14310924755807, 230.2418478263717, 230.3405864051853, 230.43932498399892, 230.53806356281254, 230.63680214162613, 230.73554072043976, 230.83427929925338, 230.93301787806698, 231.0317564568806, 231.13049503569422, 231.22923361450782, 231.32797219332144, 231.42671077213504, 231.52544935094866, 231.6241879297623, 231.72292650857588, 231.8216650873895, 231.92040366620313, 232.01914224501672, 232.11788082383035, 232.21661940264397, 232.31535798145757, 232.4140965602712, 232.5128351390848, 232.6115737178984, 232.71031229671203, 232.80905087552563, 232.90778945433925, 233.00652803315288, 233.10526661196647, 233.2040051907801, 233.30274376959372, 233.4014823484073, 233.50022092722094, 233.59895950603456, 233.69769808484816, 233.79643666366178, 233.8951752424754, 233.993913821289, 234.09265240010262, 234.19139097891622, 234.29012955772984, 234.38886813654346, 234.48760671535706, 234.58634529417068, 234.6850838729843, 234.7838224517979, 234.88256103061153, 234.98129960942515, 235.08003818823875, 235.17877676705237, 235.277515345866, 235.3762539246796, 235.4749925034932, 235.5737310823068, 235.67246966112043, 235.77120823993405, 235.86994681874765, 235.96868539756127, 236.0674239763749, 236.1661625551885, 236.26490113400212, 236.36363971281574, 236.46237829162934, 236.56111687044296, 236.65985544925658, 236.75859402807018, 236.8573326068838, 236.9560711856974, 237.05480976451102, 237.15354834332464, 237.25228692213824, 237.35102550095186, 237.4497640797655, 237.54850265857908, 237.6472412373927, 237.74597981620633, 237.84471839501992, 237.94345697383355, 238.04219555264717, 238.14093413146077, 238.2396727102744, 238.338411289088, 238.4371498679016, 238.53588844671523, 238.63462702552883, 238.73336560434245, 238.83210418315608, 238.93084276196967, 239.0295813407833, 239.12831991959692, 239.2270584984105, 239.42453565603776, 239.52327423485136, 239.62201281366498, 239.72075139247858, 239.8194899712922, 239.91822855010582, 240.01696712891942, 240.11570570773304, 240.21444428654667, 240.31318286536026, 240.41192144417388, 240.5106600229875, 240.6093986018011, 240.70813718061473, 240.80687575942835, 240.90561433824195, 241.00435291705557, 241.10309149586917, 241.2018300746828, 241.3005686534964, 241.39930723231, 241.49804581112363, 241.59678438993726, 241.69552296875085, 241.79426154756447, 241.8930001263781, 241.9917387051917, 242.09047728400532, 242.1892158628189, 242.28795444163254, 242.38669302044616, 242.48543159925975, 242.58417017807338, 242.682908756887, 242.7816473357006, 242.88038591451422, 242.97912449332784, 243.07786307214144, 243.17660165095506, 243.2753402297687, 243.37407880858228, 243.4728173873959, 243.5715559662095, 243.67029454502313, 243.76903312383675, 243.86777170265034, 243.96651028146397, 244.0652488602776, 244.1639874390912, 244.2627260179048, 244.36146459671843, 244.46020317553203, 244.55894175434565, 244.65768033315928, 244.75641891197287, 244.85515749078647, 244.95389606960012, 245.05263464841372, 245.1513732272273, 245.25011180604096, 245.34885038485456, 245.44758896366815, 245.5463275424818, 245.6450661212954, 245.743804700109, 245.84254327892265, 245.94128185773624, 246.04002043654984, 246.1387590153635, 246.23749759417709, 246.33623617299068, 246.43497475180433, 246.53371333061793, 246.63245190943152, 246.73119048824518, 246.82992906705877, 246.92866764587237, 247.02740622468596, 247.1261448034996, 247.2248833823132, 247.3236219611268, 247.42236053994046, 247.52109911875405, 247.61983769756765, 247.7185762763813, 247.8173148551949, 247.9160534340085, 248.01479201282214, 248.11353059163574, 248.21226917044933, 248.31100774926298, 248.40974632807658, 248.50848490689017, 248.60722348570383, 248.70596206451742, 248.80470064333102, 248.90343922214467, 249.00217780095826, 249.10091637977186, 249.1996549585855, 249.2983935373991, 249.3971321162127, 249.49587069502635, 249.59460927383995, 249.69334785265355, 249.79208643146714, 249.8908250102808, 249.9895635890944, 250.08830216790798, 250.18704074672164, 250.28577932553523, 250.38451790434883, 250.48325648316248, 250.58199506197607, 250.68073364078967, 250.77947221960332, 250.87821079841692, 250.9769493772305, 251.07568795604416, 251.17442653485776, 251.27316511367135, 251.371903692485, 251.4706422712986, 251.5693808501122, 251.66811942892585, 251.76685800773944, 251.86559658655304, 251.9643351653667, 252.0630737441803, 252.16181232299388, 252.26055090180753, 252.35928948062113, 252.45802805943472, 252.55676663824832, 252.65550521706197, 252.75424379587557, 252.85298237468916, 252.95172095350281, 253.0504595323164, 253.14919811113, 253.24793668994366, 253.34667526875725, 253.44541384757085, 253.5441524263845, 253.6428910051981, 253.7416295840117, 253.84036816282534, 253.93910674163894, 254.03784532045253, 254.13658389926618, 254.23532247807978, 254.33406105689338, 254.43279963570703, 254.53153821452062, 254.63027679333422, 254.72901537214787, 254.82775395096147, 254.92649252977506, 255.0252311085887, 255.1239696874023, 255.2227082662159, 255.3214468450295, 255.42018542384315, 255.51892400265675, 255.61766258147034, 255.716401160284, 255.8151397390976, 255.91387831791118, 256.0126168967248, 256.1113554755384, 256.210094054352, 256.30883263316565, 256.40757121197925, 256.50630979079284, 256.6050483696065, 256.7037869484201, 256.8025255272337, 256.90126410604734, 257.00000268486093, 257.0987412636745, 257.1974798424882, 257.2962184213018, 257.39495700011537, 257.493695578929, 257.5924341577426, 257.6911727365562, 257.78991131536986, 257.88864989418346, 257.98738847299705, 258.08612705181065, 258.1848656306243, 258.2836042094379, 258.3823427882515, 258.48108136706514, 258.57981994587874, 258.67855852469233, 258.777297103506, 258.8760356823196, 258.9747742611332, 259.07351283994683, 259.1722514187604, 259.270989997574, 259.3697285763877, 259.46846715520127, 259.56720573401486, 259.6659443128285, 259.7646828916421, 259.8634214704557, 259.96216004926936, 260.06089862808295, 260.15963720689655, 260.2583757857102, 260.3571143645238, 260.4558529433374, 260.55459152215104, 260.65333010096464, 260.75206867977823, 260.85080725859183, 260.9495458374055, 261.0482844162191, 261.14702299503267, 261.2457615738463, 261.3445001526599, 261.4432387314735, 261.54197731028717, 261.64071588910076, 261.73945446791436, 261.838193046728, 261.9369316255416, 262.0356702043552, 262.13440878316885, 262.23314736198245, 262.33188594079604, 262.4306245196097, 262.5293630984233, 262.6281016772369, 262.72684025605054, 262.82557883486413, 262.9243174136777, 263.0230559924914, 263.121794571305, 263.22053315011857, 263.3192717289322, 263.4180103077458, 263.5167488865594, 263.615487465373, 263.71422604418666, 263.81296462300025, 263.91170320181385, 264.0104417806275, 264.1091803594411, 264.2079189382547, 264.30665751706834, 264.40539609588194, 264.50413467469554, 264.6028732535092, 264.7016118323228, 264.8003504111364, 264.89908898995003, 264.9978275687636, 265.0965661475772, 265.1953047263909, 265.29404330520447, 265.39278188401806, 265.4915204628317, 265.5902590416453, 265.6889976204589, 265.78773619927256, 265.88647477808615, 265.98521335689975, 266.0839519357134, 266.182690514527, 266.2814290933406, 266.3801676721542, 266.47890625096784, 266.57764482978143, 266.67638340859503, 266.7751219874087, 266.8738605662223, 266.9725991450359, 267.0713377238495, 267.1700763026631, 267.2688148814767, 267.36755346029037, 267.46629203910396, 267.56503061791756, 267.6637691967312, 267.7625077755448, 267.8612463543584, 267.95998493317205, 268.05872351198565, 268.15746209079924, 268.2562006696129, 268.3549392484265, 268.4536778272401, 268.55241640605374, 268.65115498486733, 268.7498935636809, 268.8486321424945, 268.9473707213082, 269.04610930012177, 269.14484787893537, 269.243586457749, 269.3423250365626, 269.4410636153762, 269.53980219418986, 269.63854077300346, 269.73727935181705, 269.8360179306307, 269.9347565094443, 270.0334950882579, 270.13223366707155, 270.23097224588514, 270.32971082469874, 270.4284494035124, 270.527187982326, 270.6259265611396, 270.72466513995323, 270.8234037187668, 270.9221422975804, 271.0208808763941, 271.11961945520767, 271.21835803402126, 271.3170966128349, 271.4158351916485, 271.5145737704621, 271.6133123492757, 271.71205092808935, 271.81078950690295, 271.90952808571654, 272.0082666645302, 272.1070052433438, 272.2057438221574, 272.30448240097104, 272.40322097978463, 272.50195955859823, 272.6006981374119, 272.6994367162255, 272.7981752950391, 272.8969138738527, 272.9956524526663, 273.0943910314799, 273.19312961029357, 273.29186818910716, 273.39060676792076, 273.4893453467344, 273.588083925548, 273.6868225043616, 273.78556108317525, 273.88429966198885, 273.98303824080244, 274.0817768196161, 274.1805153984297, 274.2792539772433, 274.3779925560569, 274.47673113487053, 274.57546971368413, 274.6742082924977, 274.7729468713114, 274.87168545012497, 274.97042402893857, 275.0691626077522, 275.1679011865658, 275.2666397653794, 275.36537834419306, 275.46411692300666, 275.56285550182025, 275.6615940806339, 275.7603326594475, 275.8590712382611, 275.95780981707475, 276.05654839588834, 276.15528697470194, 276.2540255535156, 276.3527641323292, 276.4515027111428, 276.55024128995643, 276.64897986877, 276.7477184475836, 276.8464570263973, 276.94519560521087, 277.04393418402447, 277.14267276283806, 277.2414113416517, 277.3401499204653, 277.4388884992789, 277.53762707809256, 277.63636565690615, 277.73510423571975, 277.8338428145334, 277.932581393347, 278.0313199721606, 278.22879712978784, 278.32753570860143, 278.4262742874151, 278.5250128662287, 278.6237514450423, 278.7224900238559, 278.8212286026695, 278.9199671814831, 279.01870576029677, 279.11744433911036, 279.21618291792396, 279.3149214967376, 279.4136600755512, 279.5123986543648, 279.61113723317845, 279.70987581199205, 279.80861439080564, 279.90735296961924, 280.0060915484329, 280.1048301272465, 280.2035687060601, 280.30230728487373, 280.40104586368733, 280.4997844425009, 280.5985230213146, 280.6972616001282, 280.79600017894177, 280.8947387577554, 280.993477336569, 281.0922159153826, 281.19095449419626, 281.28969307300986, 281.38843165182345, 281.4871702306371, 281.5859088094507, 281.6846473882643, 281.78338596707795, 281.88212454589154, 281.98086312470514, 282.0796017035188, 282.1783402823324, 282.277078861146, 282.37581743995963, 282.4745560187732, 282.5732945975868, 282.6720331764004, 282.77077175521407, 282.86951033402767, 282.96824891284126, 283.0669874916549, 283.1657260704685, 283.2644646492821, 283.36320322809576, 283.46194180690935, 283.56068038572295, 283.6594189645366, 283.7581575433502, 283.8568961221638, 283.95563470097744, 284.05437327979104, 284.15311185860463, 284.2518504374183, 284.3505890162319, 284.4493275950455, 284.5480661738591, 284.6468047526727, 284.7455433314863, 284.84428191029997, 284.94302048911356, 285.04175906792716, 285.1404976467408, 285.2392362255544, 285.337974804368, 285.4367133831816, 285.53545196199525, 285.63419054080885, 285.73292911962244, 285.8316676984361, 285.9304062772497, 286.0291448560633, 286.12788343487694, 286.22662201369053, 286.3253605925041, 286.4240991713178, 286.5228377501314, 286.62157632894497, 286.7203149077586, 286.8190534865722, 286.9177920653858, 287.01653064419946, 287.11526922301306, 287.21400780182665, 287.3127463806403, 287.4114849594539, 287.5102235382675, 287.60896211708115, 287.70770069589474, 287.80643927470834, 287.905177853522, 288.0039164323356, 288.1026550111492, 288.2013935899628, 288.30013216877643, 288.39887074759, 288.4976093264036, 288.59634790521727, 288.69508648403087, 288.79382506284446, 288.8925636416581, 288.9913022204717, 289.0900407992853, 289.18877937809896, 289.28751795691255, 289.38625653572615, 289.4849951145398, 289.5837336933534, 289.682472272167, 289.78121085098064, 289.87994942979424, 289.97868800860783, 290.0774265874215, 290.1761651662351, 290.2749037450487, 290.3736423238623, 290.4723809026759, 290.5711194814895, 290.66985806030317, 290.76859663911677, 290.86733521793036, 290.96607379674396, 291.0648123755576, 291.1635509543712, 291.2622895331848, 291.36102811199845, 291.45976669081205, 291.55850526962564, 291.6572438484393, 291.7559824272529, 291.8547210060665, 291.95345958488014, 292.05219816369373, 292.1509367425073, 292.249675321321, 292.3484139001346, 292.44715247894817, 292.5458910577618, 292.6446296365754, 292.743368215389, 292.84210679420266, 292.94084537301626, 293.03958395182985, 293.1383225306435, 293.2370611094571, 293.3357996882707, 293.4345382670843, 293.53327684589794, 293.63201542471154, 293.73075400352513, 293.8294925823388, 293.9282311611524, 294.026969739966, 294.12570831877963, 294.2244468975932, 294.3231854764068, 294.4219240552205, 294.52066263403407, 294.61940121284766, 294.7181397916613, 294.8168783704749, 294.9156169492885, 295.01435552810216, 295.11309410691575, 295.21183268572935, 295.310571264543, 295.4093098433566, 295.5080484221702, 295.60678700098384, 295.70552557979744, 295.80426415861103, 295.9030027374247, 296.0017413162383, 296.1004798950519, 296.19921847386547, 296.2979570526791, 296.3966956314927, 296.4954342103063, 296.59417278911997, 296.69291136793356, 296.79164994674716, 296.8903885255608, 296.9891271043744, 297.087865683188, 297.18660426200165, 297.28534284081525, 297.38408141962884, 297.4828199984425, 297.5815585772561, 297.6802971560697, 297.77903573488334, 297.87777431369693, 297.9765128925105, 298.0752514713242, 298.1739900501378, 298.27272862895137, 298.371467207765, 298.4702057865786, 298.5689443653922, 298.66768294420586, 298.76642152301946, 298.86516010183306, 298.96389868064665, 299.0626372594603, 299.1613758382739, 299.2601144170875, 299.35885299590115, 299.45759157471474, 299.55633015352834, 299.655068732342, 299.7538073111556, 299.8525458899692, 299.95128446878283, 300.0500230475964, 300.14876162641, 300.2475002052237, 300.34623878403727, 300.44497736285086, 300.5437159416645, 300.6424545204781, 300.7411930992917, 300.83993167810536, 300.93867025691895, 301.03740883573255, 301.1361474145462, 301.2348859933598, 301.3336245721734, 301.43236315098704, 301.53110172980064, 301.62984030861423, 301.72857888742783, 301.8273174662415, 301.9260560450551, 302.0247946238687, 302.1235332026823, 302.2222717814959, 302.3210103603095, 302.41974893912317, 302.51848751793676, 302.61722609675036, 302.715964675564, 302.8147032543776, 302.9134418331912, 303.01218041200485, 303.11091899081845, 303.20965756963204, 303.3083961484457, 303.4071347272593, 303.5058733060729, 303.60461188488654, 303.70335046370013, 303.8020890425137, 303.9008276213274, 303.999566200141, 304.09830477895457, 304.1970433577682, 304.2957819365818, 304.3945205153954, 304.493259094209, 304.59199767302266, 304.69073625183626, 304.78947483064985, 304.8882134094635, 304.9869519882771, 305.0856905670907, 305.18442914590435, 305.28316772471794, 305.38190630353154, 305.4806448823452, 305.5793834611588, 305.6781220399724, 305.77686061878603, 305.8755991975996, 305.9743377764132, 306.0730763552269, 306.17181493404047, 306.27055351285406, 306.3692920916677, 306.4680306704813, 306.5667692492949, 306.66550782810856, 306.76424640692215, 306.86298498573575, 306.9617235645494, 307.060462143363, 307.1592007221766, 307.2579393009902, 307.35667787980384, 307.45541645861744, 307.55415503743103, 307.6528936162447, 307.7516321950583, 307.8503707738719, 307.9491093526855, 308.0478479314991, 308.1465865103127, 308.24532508912637, 308.34406366793996, 308.44280224675356, 308.5415408255672, 308.6402794043808, 308.7390179831944, 308.83775656200805, 308.93649514082165, 309.03523371963524, 309.1339722984489, 309.2327108772625, 309.3314494560761, 309.43018803488974, 309.52892661370333, 309.62766519251693, 309.7264037713306, 309.8251423501442, 309.9238809289578, 310.02261950777137, 310.121358086585, 310.2200966653986, 310.3188352442122, 310.41757382302586, 310.51631240183946, 310.61505098065305, 310.7137895594667, 310.8125281382803, 310.9112667170939, 311.01000529590755, 311.10874387472114, 311.20748245353474, 311.3062210323484, 311.404959611162, 311.5036981899756, 311.60243676878923, 311.7011753476028, 311.7999139264164, 311.8986525052301, 311.99739108404367, 312.09612966285727, 312.1948682416709, 312.2936068204845, 312.3923453992981, 312.49108397811176, 312.58982255692536, 312.68856113573895, 312.78729971455255, 312.8860382933662, 312.9847768721798, 313.0835154509934, 313.18225402980704, 313.28099260862064, 313.37973118743423, 313.4784697662479, 313.5772083450615, 313.6759469238751, 313.7746855026887, 313.8734240815023, 313.9721626603159, 314.07090123912957, 314.16963981794316, 314.26837839675676, 314.3671169755704, 314.465855554384, 314.5645941331976, 314.66333271201125, 314.76207129082485, 314.86080986963844, 314.9595484484521, 315.0582870272657, 315.1570256060793, 315.25576418489294, 315.35450276370653, 315.45324134252013, 315.5519799213337, 315.6507185001474, 315.749457078961, 315.84819565777457, 315.9469342365882, 316.0456728154018, 316.1444113942154, 316.24314997302906, 316.34188855184266, 316.44062713065625, 316.5393657094699, 316.6381042882835, 316.7368428670971, 316.83558144591075, 316.93432002472434, 317.03305860353794, 317.1317971823516, 317.2305357611652, 317.3292743399788, 317.42801291879243, 317.526751497606, 317.6254900764196, 317.7242286552333, 317.82296723404687, 317.92170581286047, 318.02044439167406, 318.1191829704877, 318.2179215493013, 318.3166601281149, 318.41539870692856, 318.51413728574215, 318.61287586455575, 318.7116144433694, 318.810353022183, 318.9090916009966, 319.00783017981024, 319.10656875862384, 319.20530733743743, 319.3040459162511, 319.4027844950647, 319.5015230738783, 319.6002616526919, 319.6990002315055, 319.7977388103191, 319.89647738913277, 319.99521596794636, 320.09395454675996, 320.1926931255736, 320.2914317043872, 320.3901702832008, 320.48890886201445, 320.58764744082805, 320.68638601964165, 320.78512459845524, 320.8838631772689, 320.9826017560825, 321.0813403348961, 321.18007891370974, 321.27881749252333, 321.3775560713369, 321.4762946501506, 321.5750332289642, 321.67377180777777, 321.7725103865914, 321.871248965405, 321.9699875442186, 322.06872612303226, 322.16746470184586, 322.26620328065945, 322.3649418594731, 322.4636804382867, 322.5624190171003, 322.66115759591395, 322.75989617472754, 322.85863475354114, 322.9573733323548, 323.0561119111684, 323.154850489982, 323.25358906879563, 323.35232764760923, 323.4510662264228, 323.5498048052364, 323.6485433840501, 323.74728196286367, 323.84602054167726, 323.9447591204909, 324.0434976993045, 324.1422362781181, 324.24097485693176, 324.33971343574535, 324.43845201455895, 324.5371905933726, 324.6359291721862, 324.7346677509998, 324.83340632981344, 324.93214490862704, 325.03088348744063, 325.1296220662543, 325.2283606450679, 325.3270992238815, 325.4258378026951, 325.5245763815087, 325.6233149603223, 325.72205353913597, 325.82079211794957, 325.91953069676316, 326.0182692755768, 326.1170078543904, 326.215746433204, 326.3144850120176, 326.41322359083125, 326.51196216964485, 326.61070074845844, 326.7094393272721, 326.8081779060857, 326.9069164848993, 327.00565506371294, 327.10439364252653, 327.2031322213401, 327.3018708001538, 327.4006093789674, 327.49934795778097, 327.5980865365946, 327.6968251154082, 327.7955636942218, 327.89430227303546, 327.99304085184906, 328.09177943066265, 328.1905180094763, 328.2892565882899, 328.3879951671035, 328.48673374591715, 328.58547232473074, 328.68421090354434, 328.782949482358, 328.8816880611716, 328.9804266399852, 329.0791652187988, 329.17790379761243, 329.276642376426, 329.3753809552396, 329.4741195340533, 329.57285811286687, 329.67159669168046, 329.7703352704941, 329.8690738493077, 329.9678124281213, 330.06655100693496, 330.16528958574855, 330.26402816456215, 330.3627667433758, 330.4615053221894, 330.560243901003, 330.65898247981664, 330.75772105863024, 330.85645963744383, 330.9551982162575, 331.0539367950711, 331.1526753738847, 331.25141395269833, 331.3501525315119, 331.4488911103255, 331.54762968913917, 331.64636826795277, 331.74510684676636, 331.84384542557996, 331.9425840043936, 332.0413225832072, 332.1400611620208, 332.23879974083445, 332.33753831964805, 332.43627689846164, 332.5350154772753, 332.6337540560889, 332.7324926349025, 332.83123121371614, 332.92996979252973, 333.0287083713433, 333.127446950157, 333.2261855289706, 333.32492410778417, 333.4236626865978, 333.5224012654114, 333.621139844225, 333.71987842303867, 333.81861700185226, 333.91735558066586, 334.0160941594795, 334.1148327382931, 334.2135713171067, 334.31230989592035, 334.41104847473395, 334.50978705354754, 334.60852563236114, 334.7072642111748, 334.8060027899884, 334.904741368802, 335.00347994761563, 335.1022185264292, 335.2009571052428, 335.2996956840565, 335.39843426287007, 335.49717284168366, 335.5959114204973, 335.6946499993109, 335.7933885781245, 335.89212715693816, 335.99086573575175, 336.08960431456535, 336.188342893379, 336.2870814721926, 336.3858200510062, 336.48455862981984, 336.58329720863344, 336.68203578744703, 336.7807743662607, 336.8795129450743, 336.9782515238879, 337.07699010270153, 337.1757286815151, 337.2744672603287, 337.3732058391423, 337.47194441795597, 337.57068299676956, 337.66942157558316, 337.7681601543968, 337.8668987332104, 337.965637312024, 338.06437589083765, 338.16311446965125, 338.26185304846484, 338.3605916272785, 338.4593302060921, 338.5580687849057, 338.65680736371934, 338.75554594253293, 338.85428452134653, 338.9530231001602, 339.0517616789738, 339.15050025778737, 339.249238836601, 339.3479774154146, 339.4467159942282, 339.54545457304187, 339.64419315185546, 339.74293173066906, 339.8416703094827, 339.9404088882963, 340.0391474671099, 340.1378860459235, 340.23662462473715, 340.33536320355074, 340.43410178236434, 340.532840361178, 340.6315789399916, 340.7303175188052, 340.82905609761883, 340.9277946764324, 341.026533255246, 341.1252718340597, 341.22401041287327, 341.32274899168686, 341.4214875705005, 341.5202261493141, 341.6189647281277, 341.71770330694136, 341.81644188575495, 341.91518046456855, 342.0139190433822, 342.1126576221958, 342.2113962010094, 342.31013477982304, 342.40887335863664, 342.50761193745024, 342.6063505162639, 342.7050890950775, 342.8038276738911, 342.9025662527047, 343.0013048315183, 343.1000434103319, 343.1987819891455, 343.29752056795917, 343.39625914677276, 343.49499772558636, 343.5937363044, 343.6924748832136, 343.7912134620272, 343.88995204084085, 343.98869061965445, 344.08742919846804, 344.1861677772817, 344.2849063560953, 344.3836449349089, 344.48238351372254, 344.58112209253613, 344.67986067134973, 344.7785992501634, 344.877337828977, 344.9760764077906, 345.0748149866042, 345.1735535654178, 345.2722921442314, 345.371030723045, 345.46976930185866, 345.56850788067226, 345.66724645948585, 345.7659850382995, 345.8647236171131, 345.9634621959267, 346.06220077474035, 346.16093935355394, 346.25967793236754, 346.3584165111812, 346.4571550899948, 346.5558936688084, 346.65463224762203, 346.7533708264356, 346.8521094052492, 346.9508479840629, 347.04958656287647, 347.14832514169007, 347.2470637205037, 347.3458022993173, 347.4445408781309, 347.54327945694456, 347.64201803575816, 347.74075661457175, 347.8394951933854, 347.938233772199, 348.0369723510126, 348.1357109298262, 348.23444950863984, 348.33318808745344, 348.43192666626703, 348.5306652450807, 348.6294038238943, 348.7281424027079, 348.8268809815215, 348.9256195603351, 349.0243581391487, 349.12309671796237, 349.22183529677596, 349.32057387558956, 349.4193124544032, 349.5180510332168, 349.6167896120304, 349.71552819084405, 349.81426676965765, 349.91300534847124, 350.0117439272849, 350.1104825060985, 350.2092210849121, 350.30795966372574, 350.40669824253933, 350.50543682135293, 350.6041754001666, 350.7029139789802, 350.8016525577938, 350.90039113660737, 350.999129715421, 351.0978682942346, 351.1966068730482, 351.29534545186186, 351.39408403067546, 351.49282260948905, 351.5915611883027, 351.6902997671163, 351.7890383459299, 351.88777692474355, 351.98651550355714, 352.08525408237074, 352.1839926611844, 352.282731239998, 352.3814698188116, 352.48020839762523, 352.57894697643883, 352.6776855552524, 352.7764241340661, 352.87516271287967, 352.97390129169327, 353.0726398705069, 353.1713784493205, 353.2701170281341, 353.36885560694776, 353.46759418576136, 353.56633276457495, 353.66507134338855, 353.7638099222022, 353.8625485010158, 353.9612870798294, 354.06002565864304, 354.15876423745664, 354.25750281627023, 354.3562413950839, 354.4549799738975, 354.5537185527111, 354.6524571315247, 354.7511957103383, 354.8499342891519, 354.94867286796557, 355.04741144677917, 355.14615002559276, 355.2448886044064, 355.34362718322, 355.4423657620336, 355.54110434084726, 355.63984291966085, 355.73858149847445, 355.8373200772881, 355.9360586561017, 356.0347972349153, 356.13353581372894, 356.23227439254254, 356.33101297135613, 356.4297515501697, 356.5284901289834, 356.627228707797, 356.72596728661057, 356.8247058654242, 356.9234444442378, 357.0221830230514, 357.12092160186506, 357.21966018067866, 357.31839875949225, 357.4171373383059, 357.5158759171195, 357.6146144959331, 357.71335307474675, 357.81209165356034, 357.91083023237394, 358.0095688111876, 358.1083073900012, 358.2070459688148, 358.30578454762843, 358.40452312644203, 358.5032617052556, 358.6020002840693, 358.7007388628829, 358.79947744169647, 358.8982160205101, 358.9969545993237, 359.0956931781373, 359.1944317569509, 359.29317033576456, 359.39190891457815, 359.49064749339175, 359.5893860722054, 359.688124651019, 359.7868632298326, 359.88560180864624, 359.98434038745984, 360.08307896627343, 360.1818175450871, 360.2805561239007, 360.3792947027143, 360.4780332815279, 360.5767718603415, 360.6755104391551, 360.77424901796877, 360.87298759678237, 360.97172617559596, 361.0704647544096, 361.1692033332232, 361.2679419120368, 361.36668049085046, 361.46541906966405, 361.56415764847765, 361.6628962272913, 361.7616348061049, 361.8603733849185, 361.9591119637321, 362.05785054254574, 362.15658912135933, 362.2553277001729, 362.3540662789866, 362.4528048578002, 362.55154343661377, 362.6502820154274, 362.749020594241, 362.8477591730546, 362.94649775186826, 363.04523633068186, 363.14397490949545, 363.2427134883091, 363.3414520671227, 363.4401906459363, 363.53892922474995, 363.63766780356355, 363.73640638237714, 363.8351449611908, 363.9338835400044, 364.032622118818, 364.13136069763164, 364.23009927644523, 364.3288378552588, 364.4275764340725, 364.5263150128861, 364.62505359169967, 364.72379217051326, 364.8225307493269, 364.9212693281405, 365.0200079069541, 365.11874648576776, 365.21748506458135, 365.31622364339495, 365.4149622222086, 365.5137008010222, 365.6124393798358, 365.71117795864944, 365.80991653746304, 365.90865511627663, 366.0073936950903, 366.1061322739039, 366.2048708527175, 366.30360943153113, 366.4023480103447, 366.5010865891583, 366.59982516797197, 366.69856374678557, 366.79730232559916, 366.8960409044128, 366.9947794832264, 367.09351806204, 367.19225664085366, 367.29099521966725, 367.38973379848085, 367.48847237729444, 367.5872109561081, 367.6859495349217, 367.7846881137353, 367.88342669254894, 367.98216527136253, 368.0809038501761, 368.1796424289898, 368.2783810078034, 368.37711958661697, 368.4758581654306, 368.5745967442442, 368.6733353230578, 368.77207390187147, 368.87081248068506, 368.96955105949866, 369.0682896383123, 369.1670282171259, 369.2657667959395, 369.36450537475315, 369.46324395356675, 369.56198253238034, 369.660721111194, 369.7594596900076, 369.8581982688212, 369.9569368476348, 370.05567542644843, 370.154414005262, 370.2531525840756, 370.3518911628893, 370.45062974170287, 370.54936832051646, 370.6481068993301, 370.7468454781437, 370.8455840569573, 370.94432263577096, 371.04306121458455, 371.14179979339815, 371.2405383722118, 371.3392769510254, 371.438015529839, 371.53675410865264, 371.63549268746624, 371.73423126627983, 371.8329698450935, 371.9317084239071, 372.0304470027207, 372.12918558153433, 372.2279241603479, 372.3266627391615, 372.4254013179752, 372.52413989678877, 372.62287847560236, 372.72161705441596, 372.8203556332296, 372.9190942120432, 373.0178327908568, 373.11657136967045, 373.21530994848405, 373.31404852729764, 373.4127871061113, 373.5115256849249, 373.6102642637385, 373.70900284255214, 373.80774142136573, 373.90648000017933, 374.005218578993, 374.1039571578066, 374.20269573662017, 374.3014343154338, 374.4001728942474, 374.498911473061, 374.59765005187467, 374.69638863068826, 374.79512720950186, 374.8938657883155, 374.9926043671291, 375.0913429459427, 375.19008152475635, 375.28882010356995, 375.38755868238354, 375.48629726119714, 375.5850358400108, 375.6837744188244, 375.782512997638, 375.88125157645163, 375.9799901552652, 376.0787287340788, 376.1774673128925, 376.27620589170607, 376.37494447051967, 376.4736830493333, 376.5724216281469, 376.6711602069605, 376.76989878577416, 376.86863736458776, 376.96737594340135, 377.066114522215, 377.1648531010286, 377.2635916798422, 377.36233025865585, 377.46106883746944, 377.55980741628304, 377.6585459950967, 377.7572845739103, 377.8560231527239, 377.95476173153753, 378.0535003103511, 378.1522388891647, 378.2509774679783, 378.34971604679197, 378.44845462560556, 378.54719320441916, 378.6459317832328, 378.7446703620464, 378.84340894086, 378.94214751967365, 379.04088609848725, 379.13962467730084, 379.2383632561145, 379.3371018349281, 379.4358404137417, 379.53457899255534, 379.63331757136893, 379.73205615018253, 379.8307947289962, 379.9295333078098, 380.0282718866234, 380.127010465437, 380.2257490442506, 380.3244876230642, 380.42322620187787, 380.52196478069146, 380.62070335950506, 380.7194419383187, 380.8181805171323, 380.9169190959459, 381.0156576747595, 381.11439625357315, 381.21313483238674, 381.31187341120034, 381.410611990014, 381.5093505688276, 381.6080891476412, 381.70682772645483, 381.8055663052684, 381.904304884082, 382.0030434628957, 382.10178204170927, 382.20052062052287, 382.2992591993365, 382.3979977781501, 382.4967363569637, 382.59547493577736, 382.69421351459096, 382.79295209340455, 382.8916906722182, 382.9904292510318, 383.0891678298454, 383.18790640865905, 383.28664498747264, 383.38538356628624, 383.4841221450999, 383.5828607239135, 383.6815993027271, 383.7803378815407, 383.8790764603543, 383.9778150391679, 384.0765536179815, 384.17529219679517, 384.27403077560876, 384.37276935442236, 384.471507933236, 384.5702465120496, 384.6689850908632, 384.76772366967685, 384.86646224849045, 384.96520082730405, 385.0639394061177, 385.1626779849313, 385.2614165637449, 385.36015514255854, 385.45889372137214, 385.55763230018573, 385.6563708789994, 385.755109457813, 385.8538480366266, 385.9525866154402, 386.0513251942538, 386.1500637730674, 386.24880235188107, 386.34754093069466, 386.44627950950826, 386.54501808832185, 386.6437566671355, 386.7424952459491, 386.8412338247627, 386.93997240357635, 387.03871098238994, 387.13744956120354, 387.2361881400172, 387.3349267188308, 387.4336652976444, 387.53240387645803, 387.63114245527163, 387.7298810340852, 387.8286196128989, 387.9273581917125, 388.02609677052607, 388.1248353493397, 388.2235739281533, 388.3223125069669, 388.42105108578056, 388.51978966459416, 388.61852824340775, 388.7172668222214, 388.816005401035, 388.9147439798486, 389.01348255866225, 389.11222113747584, 389.21095971628944, 389.30969829510303, 389.4084368739167, 389.5071754527303, 389.6059140315439, 389.7046526103575, 389.8033911891711, 389.9021297679847, 390.00086834679837, 390.09960692561197, 390.19834550442556, 390.2970840832392, 390.3958226620528, 390.4945612408664, 390.59329981968006, 390.69203839849365, 390.79077697730725, 390.8895155561209, 390.9882541349345, 391.0869927137481, 391.18573129256174, 391.28446987137534, 391.38320845018893, 391.4819470290026, 391.5806856078162, 391.6794241866298, 391.7781627654434, 391.876901344257, 391.9756399230706, 392.0743785018842, 392.17311708069786, 392.27185565951146, 392.37059423832505, 392.4693328171387, 392.5680713959523, 392.6668099747659, 392.76554855357955, 392.86428713239314, 392.96302571120674, 393.0617642900204, 393.160502868834, 393.2592414476476, 393.35798002646123, 393.45671860527483, 393.5554571840884, 393.6541957629021, 393.7529343417157, 393.85167292052927, 393.9504114993429, 394.0491500781565, 394.1478886569701, 394.24662723578376, 394.34536581459736, 394.44410439341095, 394.54284297222455, 394.6415815510382, 394.7403201298518, 394.8390587086654, 394.93779728747904, 395.03653586629264, 395.13527444510623, 395.2340130239199, 395.3327516027335, 395.4314901815471, 395.5302287603607, 395.6289673391743, 395.7277059179879, 395.82644449680157, 395.92518307561517, 396.02392165442876, 396.1226602332424, 396.221398812056, 396.3201373908696, 396.41887596968326, 396.51761454849685, 396.61635312731045, 396.7150917061241, 396.8138302849377, 396.9125688637513, 397.01130744256494, 397.11004602137854, 397.20878460019213, 397.3075231790057, 397.4062617578194, 397.505000336633, 397.60373891544657, 397.7024774942602, 397.8012160730738, 397.8999546518874, 397.99869323070106, 398.09743180951466, 398.19617038832826, 398.2949089671419, 398.3936475459555, 398.4923861247691, 398.59112470358275, 398.68986328239635, 398.78860186120994, 398.8873404400236, 398.9860790188372, 399.0848175976508, 399.18355617646444, 399.28229475527803, 399.3810333340916, 399.4797719129053, 399.5785104917189, 399.67724907053247, 399.7759876493461, 399.8747262281597, 399.9734648069733, 400.0722033857869, 400.17094196460056, 400.26968054341415, 400.36841912222775, 400.4671577010414, 400.565896279855, 400.6646348586686, 400.76337343748224, 400.86211201629584, 400.96085059510943, 401.0595891739231, 401.1583277527367, 401.2570663315503, 401.35580491036393, 401.4545434891775, 401.5532820679911, 401.6520206468048, 401.75075922561837, 401.84949780443196, 401.9482363832456, 402.0469749620592, 402.1457135408728, 402.24445211968646, 402.34319069850005, 402.44192927731365, 402.5406678561273, 402.6394064349409, 402.7381450137545, 402.8368835925681, 402.93562217138174, 403.03436075019533, 403.1330993290089, 403.2318379078226, 403.3305764866362, 403.42931506544977, 403.5280536442634, 403.626792223077, 403.7255308018906, 403.82426938070427, 403.92300795951786, 404.02174653833146, 404.1204851171451, 404.2192236959587, 404.3179622747723, 404.41670085358595, 404.51543943239955, 404.61417801121314, 404.7129165900268, 404.8116551688404, 404.910393747654, 405.00913232646764, 405.10787090528123, 405.2066094840948, 405.3053480629085, 405.4040866417221, 405.50282522053567, 405.60156379934926, 405.7003023781629, 405.7990409569765, 405.8977795357901, 405.99651811460376, 406.09525669341735, 406.19399527223095, 406.2927338510446, 406.3914724298582, 406.4902110086718, 406.58894958748544, 406.68768816629904, 406.78642674511264, 406.8851653239263, 406.9839039027399, 407.0826424815535, 407.18138106036713, 407.2801196391807, 407.3788582179943, 407.477596796808, 407.57633537562157, 407.67507395443516, 407.7738125332488, 407.8725511120624, 407.971289690876, 408.07002826968966, 408.16876684850325, 408.26750542731685, 408.36624400613044, 408.4649825849441, 408.5637211637577, 408.6624597425713, 408.76119832138494, 408.85993690019853, 408.95867547901213, 409.0574140578258, 409.1561526366394, 409.254891215453, 409.3536297942666, 409.4523683730802, 409.5511069518938, 409.64984553070747, 409.74858410952106, 409.84732268833466, 409.9460612671483, 410.0447998459619, 410.1435384247755, 410.24227700358915, 410.34101558240275, 410.43975416121634, 410.53849274003, 410.6372313188436, 410.7359698976572, 410.83470847647084, 410.93344705528443, 411.032185634098, 411.1309242129116, 411.2296627917253, 411.32840137053887, 411.42713994935247, 411.5258785281661, 411.6246171069797, 411.7233556857933, 411.82209426460696, 411.92083284342056, 412.01957142223415, 412.1183100010478, 412.2170485798614, 412.315787158675, 412.41452573748865, 412.51326431630224, 412.61200289511584, 412.7107414739295, 412.8094800527431, 412.9082186315567, 413.00695721037033, 413.1056957891839, 413.2044343679975, 413.3031729468112, 413.40191152562477, 413.50065010443836, 413.599388683252, 413.6981272620656, 413.7968658408792, 413.8956044196928, 413.99434299850645, 414.09308157732005, 414.19182015613364, 414.2905587349473, 414.3892973137609, 414.4880358925745, 414.58677447138814, 414.68551305020173, 414.78425162901533, 414.882990207829, 414.9817287866426, 415.0804673654562, 415.1792059442698, 415.2779445230834, 415.376683101897, 415.47542168071067, 415.57416025952426, 415.67289883833786, 415.7716374171515, 415.8703759959651, 415.9691145747787, 416.06785315359235, 416.16659173240595, 416.26533031121954, 416.3640688900332, 416.4628074688468, 416.5615460476604, 416.660284626474, 416.75902320528763, 416.8577617841012, 416.9565003629148, 417.0552389417285, 417.15397752054207, 417.25271609935567, 417.3514546781693, 417.4501932569829, 417.5489318357965, 417.64767041461016, 417.74640899342376, 417.84514757223735, 417.943886151051, 418.0426247298646, 418.1413633086782, 418.24010188749185, 418.33884046630544, 418.43757904511904, 418.5363176239327, 418.6350562027463, 418.7337947815599, 418.83253336037353, 418.9312719391871, 419.0300105180007], "yaxis": "y"}],
                        {"legend": {"tracegroupgap": 0}, "margin": {"t": 60}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "tickangle": 45, "tickmode": "array", "ticktext": ["2020-06-26", "2020-06-27", "2020-06-28", "2020-06-29", "2020-06-30", "2020-07-01", "2020-07-02", "2020-07-03", "2020-07-04", "2020-07-05", "2020-07-06", "2020-07-07", "2020-07-08", "2020-07-09", "2020-07-10", "2020-07-11", "2020-07-12", "2020-07-13", "2020-07-14", "2020-07-15", "2020-07-16", "2020-07-17", "2020-07-18", "2020-07-19", "2020-07-20", "2020-07-21", "2020-07-22", "2020-07-23", "2020-07-24", "2020-07-25", "2020-07-26", "2020-07-27", "2020-07-28", "2020-07-29", "2020-07-30", "2020-07-31", "2020-08-01", "2020-08-02", "2020-08-03", "2020-08-04", "2020-08-05", "2020-08-06", "2020-08-07", "2020-08-08", "2020-08-09", "2020-08-10", "2020-08-11", "2020-08-12", "2020-08-13", "2020-08-14", "2020-08-15", "2020-08-16", "2020-08-17", "2020-08-18", "2020-08-19", "2020-08-20", "2020-08-21", "2020-08-22", "2020-08-23", "2020-08-24", "2020-08-25", "2020-08-26", "2020-08-27", "2020-08-28", "2020-08-29", "2020-08-30", "2020-08-31", "2020-09-01", "2020-09-02", "2020-09-03", "2020-09-04", "2020-09-05", "2020-09-06", "2020-09-07", "2020-09-08", "2020-09-09", "2020-09-10", "2020-09-11", "2020-09-12", "2020-09-13", "2020-09-14", "2020-09-15", "2020-09-16", "2020-09-17", "2020-09-18", "2020-09-19", "2020-09-20", "2020-09-21", "2020-09-22", "2020-09-23", "2020-09-24", "2020-09-25", "2020-09-26", "2020-09-27", "2020-09-28", "2020-09-29", "2020-09-30", "2020-10-01", "2020-10-02", "2020-10-03", "2020-10-04", "2020-10-05", "2020-10-06", "2020-10-07", "2020-10-08", "2020-10-09", "2020-10-10", "2020-10-11", "2020-10-12", "2020-10-13", "2020-10-14", "2020-10-15", "2020-10-16", "2020-10-17", "2020-10-18", "2020-10-19", "2020-10-20", "2020-10-21", "2020-10-22", "2020-10-23", "2020-10-24", "2020-10-25", "2020-10-26", "2020-10-27", "2020-10-28", "2020-10-29", "2020-10-30", "2020-10-31", "2020-11-01", "2020-11-02", "2020-11-03", "2020-11-04", "2020-11-05", "2020-11-06", "2020-11-07", "2020-11-08", "2020-11-09", "2020-11-10", "2020-11-11", "2020-11-12", "2020-11-13", "2020-11-14", "2020-11-15", "2020-11-16", "2020-11-17", "2020-11-18", "2020-11-19", "2020-11-20", "2020-11-21", "2020-11-22", "2020-11-23", "2020-11-24", "2020-11-25", "2020-11-26", "2020-11-27", "2020-11-28", "2020-11-29", "2020-11-30", "2020-12-01", "2020-12-02", "2020-12-03", "2020-12-04", "2020-12-05", "2020-12-06", "2020-12-07", "2020-12-08", "2020-12-09", "2020-12-10", "2020-12-11", "2020-12-12", "2020-12-13", "2020-12-14", "2020-12-15", "2020-12-16", "2020-12-17", "2020-12-18", "2020-12-19", "2020-12-20", "2020-12-21", "2020-12-22", "2020-12-23", "2020-12-24", "2020-12-25", "2020-12-26", "2020-12-27", "2020-12-28", "2020-12-29", "2020-12-30", "2020-12-31", "2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07", "2021-01-08", "2021-01-09", "2021-01-10", "2021-01-11", "2021-01-12", "2021-01-13", "2021-01-14", "2021-01-15", "2021-01-16", "2021-01-17", "2021-01-18", "2021-01-19", "2021-01-20", "2021-01-21", "2021-01-22", "2021-01-23", "2021-01-24", "2021-01-25", "2021-01-26", "2021-01-27", "2021-01-28", "2021-01-29", "2021-01-30", "2021-01-31", "2021-02-01", "2021-02-02", "2021-02-03", "2021-02-04", "2021-02-05", "2021-02-06", "2021-02-07", "2021-02-08", "2021-02-09", "2021-02-10", "2021-02-11", "2021-02-12", "2021-02-13", "2021-02-14", "2021-02-15", "2021-02-16", "2021-02-17", "2021-02-18", "2021-02-19", "2021-02-20", "2021-02-21", "2021-02-22", "2021-02-23", "2021-02-24", "2021-02-25", "2021-02-26", "2021-02-27", "2021-02-28", "2021-03-01", "2021-03-02", "2021-03-03", "2021-03-04", "2021-03-05", "2021-03-06", "2021-03-07", "2021-03-08", "2021-03-09", "2021-03-10", "2021-03-11", "2021-03-12", "2021-03-13", "2021-03-14", "2021-03-15", "2021-03-16", "2021-03-17", "2021-03-18", "2021-03-19", "2021-03-20", "2021-03-21", "2021-03-22", "2021-03-23", "2021-03-24", "2021-03-25", "2021-03-26", "2021-03-27", "2021-03-28", "2021-03-29", "2021-03-30", "2021-03-31", "2021-04-01", "2021-04-02", "2021-04-03", "2021-04-04", "2021-04-05", "2021-04-06", "2021-04-07", "2021-04-08", "2021-04-09", "2021-04-10", "2021-04-11", "2021-04-12", "2021-04-13", "2021-04-14", "2021-04-15", "2021-04-16", "2021-04-17", "2021-04-18", "2021-04-19", "2021-04-20", "2021-04-21", "2021-04-22", "2021-04-23", "2021-04-24", "2021-04-25", "2021-04-26", "2021-04-27", "2021-04-28", "2021-04-29", "2021-04-30", "2021-05-01", "2021-05-02", "2021-05-03", "2021-05-04", "2021-05-05", "2021-05-06", "2021-05-07", "2021-05-08", "2021-05-09", "2021-05-10", "2021-05-11", "2021-05-12", "2021-05-13", "2021-05-14", "2021-05-15", "2021-05-16", "2021-05-17", "2021-05-18", "2021-05-19", "2021-05-20", "2021-05-21", "2021-05-22", "2021-05-23", "2021-05-24", "2021-05-25", "2021-05-26", "2021-05-27", "2021-05-28", "2021-05-29", "2021-05-30", "2021-05-31", "2021-06-01", "2021-06-02", "2021-06-03", "2021-06-04", "2021-06-05", "2021-06-06", "2021-06-07", "2021-06-08", "2021-06-09", "2021-06-10", "2021-06-11", "2021-06-12", "2021-06-13", "2021-06-14", "2021-06-15", "2021-06-16", "2021-06-17", "2021-06-18", "2021-06-19", "2021-06-20", "2021-06-21", "2021-06-22", "2021-06-23", "2021-06-24", "2021-06-25", "2021-06-26", "2021-06-27", "2021-06-28", "2021-06-29", "2021-06-30", "2021-07-01", "2021-07-02", "2021-07-03", "2021-07-04", "2021-07-05", "2021-07-06", "2021-07-07", "2021-07-08", "2021-07-09", "2021-07-10", "2021-07-11", "2021-07-12", "2021-07-13", "2021-07-14", "2021-07-15", "2021-07-16", "2021-07-17", "2021-07-18", "2021-07-19", "2021-07-20", "2021-07-21", "2021-07-22", "2021-07-23", "2021-07-24", "2021-07-25", "2021-07-26", "2021-07-27", "2021-07-28", "2021-07-29", "2021-07-30", "2021-07-31", "2021-08-01", "2021-08-02", "2021-08-03", "2021-08-04", "2021-08-05", "2021-08-06", "2021-08-07", "2021-08-08", "2021-08-09", "2021-08-10", "2021-08-11", "2021-08-12", "2021-08-13", "2021-08-14", "2021-08-15", "2021-08-16", "2021-08-17", "2021-08-18", "2021-08-19", "2021-08-20", "2021-08-21", "2021-08-22", "2021-08-23", "2021-08-24", "2021-08-25", "2021-08-26", "2021-08-27", "2021-08-28", "2021-08-29", "2021-08-30", "2021-08-31", "2021-09-01", "2021-09-02", "2021-09-03", "2021-09-04", "2021-09-05", "2021-09-06", "2021-09-07", "2021-09-08", "2021-09-09", "2021-09-10", "2021-09-11", "2021-09-12", "2021-09-13", "2021-09-14", "2021-09-15", "2021-09-16", "2021-09-17", "2021-09-18", "2021-09-19", "2021-09-20", "2021-09-21", "2021-09-22", "2021-09-23", "2021-09-24", "2021-09-25", "2021-09-26", "2021-09-27", "2021-09-28", "2021-09-29", "2021-09-30", "2021-10-01", "2021-10-02", "2021-10-03", "2021-10-04", "2021-10-05", "2021-10-06", "2021-10-07", "2021-10-08", "2021-10-09", "2021-10-10", "2021-10-11", "2021-10-12", "2021-10-13", "2021-10-14", "2021-10-15", "2021-10-16", "2021-10-17", "2021-10-18", "2021-10-19", "2021-10-20", "2021-10-21", "2021-10-22", "2021-10-23", "2021-10-24", "2021-10-25", "2021-10-26", "2021-10-27", "2021-10-28", "2021-10-29", "2021-10-30", "2021-10-31", "2021-11-01", "2021-11-02", "2021-11-03", "2021-11-04", "2021-11-05", "2021-11-06", "2021-11-07", "2021-11-08", "2021-11-09", "2021-11-10", "2021-11-11", "2021-11-12", "2021-11-13", "2021-11-14", "2021-11-15", "2021-11-16", "2021-11-17", "2021-11-18", "2021-11-19", "2021-11-20", "2021-11-21", "2021-11-22", "2021-11-23", "2021-11-24", "2021-11-25", "2021-11-26", "2021-11-27", "2021-11-28", "2021-11-29", "2021-11-30", "2021-12-01", "2021-12-02", "2021-12-03", "2021-12-04", "2021-12-05", "2021-12-06", "2021-12-07", "2021-12-08", "2021-12-09", "2021-12-10", "2021-12-11", "2021-12-12", "2021-12-13", "2021-12-14", "2021-12-15", "2021-12-16", "2021-12-17", "2021-12-18", "2021-12-19", "2021-12-20", "2021-12-21", "2021-12-22", "2021-12-23", "2021-12-24", "2021-12-25", "2021-12-26", "2021-12-27", "2021-12-28", "2021-12-29", "2021-12-30", "2021-12-31", "2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04", "2022-01-05", "2022-01-06", "2022-01-07", "2022-01-08", "2022-01-09", "2022-01-10", "2022-01-11", "2022-01-12", "2022-01-13", "2022-01-14", "2022-01-15", "2022-01-16", "2022-01-17", "2022-01-18", "2022-01-19", "2022-01-20", "2022-01-21", "2022-01-22", "2022-01-23", "2022-01-24", "2022-01-25", "2022-01-26", "2022-01-27", "2022-01-28", "2022-01-29", "2022-01-30", "2022-01-31", "2022-02-01", "2022-02-02", "2022-02-03", "2022-02-04", "2022-02-05", "2022-02-06", "2022-02-07", "2022-02-08", "2022-02-09", "2022-02-10", "2022-02-11", "2022-02-12", "2022-02-13", "2022-02-14", "2022-02-15", "2022-02-16", "2022-02-17", "2022-02-18", "2022-02-19", "2022-02-20", "2022-02-21", "2022-02-22", "2022-02-23", "2022-02-24", "2022-02-25", "2022-02-26", "2022-02-27", "2022-02-28", "2022-03-01", "2022-03-02", "2022-03-03", "2022-03-04", "2022-03-05", "2022-03-06", "2022-03-07", "2022-03-08", "2022-03-09", "2022-03-10", "2022-03-11", "2022-03-12", "2022-03-13", "2022-03-14", "2022-03-15", "2022-03-16", "2022-03-17", "2022-03-18", "2022-03-19", "2022-03-20", "2022-03-21", "2022-03-22", "2022-03-23", "2022-03-24", "2022-03-25", "2022-03-26", "2022-03-27", "2022-03-28", "2022-03-29", "2022-03-30", "2022-03-31", "2022-04-01", "2022-04-02", "2022-04-03", "2022-04-04", "2022-04-05", "2022-04-06", "2022-04-07", "2022-04-08", "2022-04-09", "2022-04-10", "2022-04-11", "2022-04-12", "2022-04-13", "2022-04-14", "2022-04-15", "2022-04-16", "2022-04-17", "2022-04-18", "2022-04-19", "2022-04-20", "2022-04-21", "2022-04-22", "2022-04-23", "2022-04-24", "2022-04-25", "2022-04-26", "2022-04-27", "2022-04-28", "2022-04-29", "2022-04-30", "2022-05-01", "2022-05-02", "2022-05-03", "2022-05-04", "2022-05-05", "2022-05-06", "2022-05-07", "2022-05-08", "2022-05-09", "2022-05-10", "2022-05-11", "2022-05-12", "2022-05-13", "2022-05-14", "2022-05-15", "2022-05-16", "2022-05-17", "2022-05-18", "2022-05-19", "2022-05-20", "2022-05-21", "2022-05-22", "2022-05-23", "2022-05-24", "2022-05-25", "2022-05-26", "2022-05-27", "2022-05-28", "2022-05-29", "2022-05-30", "2022-05-31", "2022-06-01", "2022-06-02", "2022-06-03", "2022-06-04", "2022-06-05", "2022-06-06", "2022-06-07", "2022-06-08", "2022-06-09", "2022-06-10", "2022-06-11", "2022-06-12", "2022-06-13", "2022-06-14", "2022-06-15", "2022-06-16", "2022-06-17", "2022-06-18", "2022-06-19", "2022-06-20", "2022-06-21", "2022-06-22", "2022-06-23", "2022-06-24", "2022-06-25", "2022-06-26", "2022-06-27", "2022-06-28", "2022-06-29", "2022-06-30", "2022-07-01", "2022-07-02", "2022-07-03", "2022-07-04", "2022-07-05", "2022-07-06", "2022-07-07", "2022-07-08", "2022-07-09", "2022-07-10", "2022-07-11", "2022-07-12", "2022-07-13", "2022-07-14", "2022-07-15", "2022-07-16", "2022-07-17", "2022-07-18", "2022-07-19", "2022-07-20", "2022-07-21", "2022-07-22", "2022-07-23", "2022-07-24", "2022-07-25", "2022-07-26", "2022-07-27", "2022-07-28", "2022-07-29", "2022-07-30", "2022-07-31", "2022-08-01", "2022-08-02", "2022-08-03", "2022-08-04", "2022-08-05", "2022-08-06", "2022-08-07", "2022-08-08", "2022-08-09", "2022-08-10", "2022-08-11", "2022-08-12", "2022-08-13", "2022-08-14", "2022-08-15", "2022-08-16", "2022-08-17", "2022-08-18", "2022-08-19", "2022-08-20", "2022-08-21", "2022-08-22", "2022-08-23", "2022-08-24", "2022-08-25", "2022-08-26", "2022-08-27", "2022-08-28", "2022-08-29", "2022-08-30", "2022-08-31", "2022-09-01", "2022-09-02", "2022-09-03", "2022-09-04", "2022-09-05", "2022-09-06", "2022-09-07", "2022-09-08", "2022-09-09", "2022-09-10", "2022-09-11", "2022-09-12", "2022-09-13", "2022-09-14", "2022-09-15", "2022-09-16", "2022-09-17", "2022-09-18", "2022-09-19", "2022-09-20", "2022-09-21", "2022-09-22", "2022-09-23", "2022-09-24", "2022-09-25", "2022-09-26", "2022-09-27", "2022-09-28", "2022-09-29", "2022-09-30", "2022-10-01", "2022-10-02", "2022-10-03", "2022-10-04", "2022-10-05", "2022-10-06", "2022-10-07", "2022-10-08", "2022-10-09", "2022-10-10", "2022-10-11", "2022-10-12", "2022-10-13", "2022-10-14", "2022-10-15", "2022-10-16", "2022-10-17", "2022-10-18", "2022-10-19", "2022-10-20", "2022-10-21", "2022-10-22", "2022-10-23", "2022-10-24", "2022-10-25", "2022-10-26", "2022-10-27", "2022-10-28", "2022-10-29", "2022-10-30", "2022-10-31", "2022-11-01", "2022-11-02", "2022-11-03", "2022-11-04", "2022-11-05", "2022-11-06", "2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10", "2022-11-11", "2022-11-12", "2022-11-13", "2022-11-14", "2022-11-15", "2022-11-16", "2022-11-17", "2022-11-18", "2022-11-19", "2022-11-20", "2022-11-21", "2022-11-22", "2022-11-23", "2022-11-24", "2022-11-25", "2022-11-26", "2022-11-27", "2022-11-28", "2022-11-29", "2022-11-30", "2022-12-01", "2022-12-02", "2022-12-03", "2022-12-04", "2022-12-05", "2022-12-06", "2022-12-07", "2022-12-08", "2022-12-09", "2022-12-10", "2022-12-11", "2022-12-12", "2022-12-13", "2022-12-14", "2022-12-15", "2022-12-16", "2022-12-17", "2022-12-18", "2022-12-19", "2022-12-20", "2022-12-21", "2022-12-22", "2022-12-23", "2022-12-24", "2022-12-25", "2022-12-26", "2022-12-27", "2022-12-28", "2022-12-29", "2022-12-30", "2022-12-31", "2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06", "2023-01-07", "2023-01-08", "2023-01-09", "2023-01-10", "2023-01-11", "2023-01-12", "2023-01-13", "2023-01-14", "2023-01-15", "2023-01-16", "2023-01-17", "2023-01-18", "2023-01-19", "2023-01-20", "2023-01-21", "2023-01-22", "2023-01-23", "2023-01-24", "2023-01-25", "2023-01-26", "2023-01-27", "2023-01-28", "2023-01-29", "2023-01-30", "2023-01-31", "2023-02-01", "2023-02-02", "2023-02-03", "2023-02-04", "2023-02-05", "2023-02-06", "2023-02-07", "2023-02-08", "2023-02-09", "2023-02-10", "2023-02-11", "2023-02-12", "2023-02-13", "2023-02-14", "2023-02-15", "2023-02-16", "2023-02-17", "2023-02-18", "2023-02-19", "2023-02-20", "2023-02-21", "2023-02-22", "2023-02-23", "2023-02-24", "2023-02-25", "2023-02-26", "2023-02-27", "2023-02-28", "2023-03-01", "2023-03-02", "2023-03-03", "2023-03-04", "2023-03-05", "2023-03-06", "2023-03-07", "2023-03-08", "2023-03-09", "2023-03-10", "2023-03-11", "2023-03-12", "2023-03-13", "2023-03-14", "2023-03-15", "2023-03-16", "2023-03-17", "2023-03-18", "2023-03-19", "2023-03-20", "2023-03-21", "2023-03-22", "2023-03-23", "2023-03-24", "2023-03-25", "2023-03-26", "2023-03-27", "2023-03-28", "2023-03-29", "2023-03-30", "2023-03-31", "2023-04-01", "2023-04-02", "2023-04-03", "2023-04-04", "2023-04-05", "2023-04-06", "2023-04-07", "2023-04-08", "2023-04-09", "2023-04-10", "2023-04-11", "2023-04-12", "2023-04-13", "2023-04-14", "2023-04-15", "2023-04-16", "2023-04-17", "2023-04-18", "2023-04-19", "2023-04-20", "2023-04-21", "2023-04-22", "2023-04-23", "2023-04-24", "2023-04-25", "2023-04-26", "2023-04-27", "2023-04-28", "2023-04-29", "2023-04-30", "2023-05-01", "2023-05-02", "2023-05-03", "2023-05-04", "2023-05-05", "2023-05-06", "2023-05-07", "2023-05-08", "2023-05-09", "2023-05-10", "2023-05-11", "2023-05-12", "2023-05-13", "2023-05-14", "2023-05-15", "2023-05-16", "2023-05-17", "2023-05-18", "2023-05-19", "2023-05-20", "2023-05-21", "2023-05-22", "2023-05-23", "2023-05-24", "2023-05-25", "2023-05-26", "2023-05-27", "2023-05-28", "2023-05-29", "2023-05-30", "2023-05-31", "2023-06-01", "2023-06-02", "2023-06-03", "2023-06-04", "2023-06-05", "2023-06-06", "2023-06-07", "2023-06-08", "2023-06-09", "2023-06-10", "2023-06-11", "2023-06-12", "2023-06-13", "2023-06-14", "2023-06-15", "2023-06-16", "2023-06-17", "2023-06-18", "2023-06-19", "2023-06-20", "2023-06-21", "2023-06-22", "2023-06-23", "2023-06-24", "2023-06-25", "2023-06-26", "2023-06-27", "2023-06-28", "2023-06-29", "2023-06-30", "2023-07-01", "2023-07-02", "2023-07-03", "2023-07-04", "2023-07-05", "2023-07-06", "2023-07-07", "2023-07-08", "2023-07-09", "2023-07-10", "2023-07-11", "2023-07-12", "2023-07-13", "2023-07-14", "2023-07-15", "2023-07-16", "2023-07-17", "2023-07-18", "2023-07-19", "2023-07-20", "2023-07-21", "2023-07-22", "2023-07-23", "2023-07-24", "2023-07-25", "2023-07-26", "2023-07-27", "2023-07-28", "2023-07-29", "2023-07-30", "2023-07-31", "2023-08-01", "2023-08-02", "2023-08-03", "2023-08-04", "2023-08-05", "2023-08-06", "2023-08-07", "2023-08-08", "2023-08-09", "2023-08-10", "2023-08-11", "2023-08-12", "2023-08-13", "2023-08-14", "2023-08-15", "2023-08-16", "2023-08-17", "2023-08-18", "2023-08-19", "2023-08-20", "2023-08-21", "2023-08-22", "2023-08-23", "2023-08-24", "2023-08-25", "2023-08-26", "2023-08-27", "2023-08-28", "2023-08-29", "2023-08-30", "2023-08-31", "2023-09-01", "2023-09-02", "2023-09-03", "2023-09-04", "2023-09-05", "2023-09-06", "2023-09-07", "2023-09-08", "2023-09-09", "2023-09-10", "2023-09-11", "2023-09-12", "2023-09-13", "2023-09-14", "2023-09-15", "2023-09-16", "2023-09-17", "2023-09-18", "2023-09-19", "2023-09-20", "2023-09-21", "2023-09-22", "2023-09-23", "2023-09-24", "2023-09-25", "2023-09-26", "2023-09-27", "2023-09-28", "2023-09-29", "2023-09-30", "2023-10-01", "2023-10-02", "2023-10-03", "2023-10-04", "2023-10-05", "2023-10-06", "2023-10-07", "2023-10-08", "2023-10-09", "2023-10-10", "2023-10-11", "2023-10-12", "2023-10-13", "2023-10-14", "2023-10-15", "2023-10-16", "2023-10-17", "2023-10-18", "2023-10-19", "2023-10-20", "2023-10-21", "2023-10-22", "2023-10-23", "2023-10-24", "2023-10-25", "2023-10-26", "2023-10-27", "2023-10-28", "2023-10-29", "2023-10-30", "2023-10-31", "2023-11-01", "2023-11-02", "2023-11-03", "2023-11-04", "2023-11-05", "2023-11-06", "2023-11-07", "2023-11-08", "2023-11-09", "2023-11-10", "2023-11-11", "2023-11-12", "2023-11-13", "2023-11-14", "2023-11-15", "2023-11-16", "2023-11-17", "2023-11-18", "2023-11-19", "2023-11-20", "2023-11-21", "2023-11-22", "2023-11-23", "2023-11-24", "2023-11-25", "2023-11-26", "2023-11-27", "2023-11-28", "2023-11-29", "2023-11-30", "2023-12-01", "2023-12-02", "2023-12-03", "2023-12-04", "2023-12-05", "2023-12-06", "2023-12-07", "2023-12-08", "2023-12-09", "2023-12-10", "2023-12-11", "2023-12-12", "2023-12-13", "2023-12-14", "2023-12-15", "2023-12-16", "2023-12-17", "2023-12-18", "2023-12-19", "2023-12-20", "2023-12-21", "2023-12-22", "2023-12-23", "2023-12-24", "2023-12-25", "2023-12-26", "2023-12-27", "2023-12-28", "2023-12-29", "2023-12-30", "2023-12-31", "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12", "2024-01-13", "2024-01-14", "2024-01-15", "2024-01-16", "2024-01-17", "2024-01-18", "2024-01-19", "2024-01-20", "2024-01-21", "2024-01-22", "2024-01-23", "2024-01-24", "2024-01-25", "2024-01-26", "2024-01-27", "2024-01-28", "2024-01-29", "2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02", "2024-02-03", "2024-02-04", "2024-02-05", "2024-02-06", "2024-02-07", "2024-02-08", "2024-02-09", "2024-02-10", "2024-02-11", "2024-02-12", "2024-02-13", "2024-02-14", "2024-02-15", "2024-02-16", "2024-02-17", "2024-02-18", "2024-02-19", "2024-02-20", "2024-02-21", "2024-02-22", "2024-02-23", "2024-02-24", "2024-02-25", "2024-02-26", "2024-02-27", "2024-02-28", "2024-02-29", "2024-03-01", "2024-03-02", "2024-03-03", "2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07", "2024-03-08", "2024-03-09", "2024-03-10", "2024-03-11", "2024-03-12", "2024-03-13", "2024-03-14", "2024-03-15", "2024-03-16", "2024-03-17", "2024-03-18", "2024-03-19", "2024-03-20", "2024-03-21", "2024-03-22", "2024-03-23", "2024-03-24", "2024-03-25", "2024-03-26", "2024-03-27", "2024-03-28", "2024-03-29", "2024-03-30", "2024-03-31", "2024-04-01", "2024-04-02", "2024-04-03", "2024-04-04", "2024-04-05", "2024-04-06", "2024-04-07", "2024-04-08", "2024-04-09", "2024-04-10", "2024-04-11", "2024-04-12", "2024-04-13", "2024-04-14", "2024-04-15", "2024-04-16", "2024-04-17", "2024-04-18", "2024-04-19", "2024-04-20", "2024-04-21", "2024-04-22", "2024-04-23", "2024-04-24", "2024-04-25", "2024-04-26", "2024-04-27", "2024-04-28", "2024-04-29", "2024-04-30", "2024-05-01", "2024-05-02", "2024-05-03", "2024-05-04", "2024-05-05", "2024-05-06", "2024-05-07", "2024-05-08", "2024-05-09", "2024-05-10", "2024-05-11", "2024-05-12", "2024-05-13", "2024-05-14", "2024-05-15", "2024-05-16", "2024-05-17", "2024-05-18", "2024-05-19", "2024-05-20", "2024-05-21", "2024-05-22", "2024-05-23", "2024-05-24", "2024-05-25", "2024-05-26", "2024-05-27", "2024-05-28", "2024-05-29", "2024-05-30", "2024-05-31", "2024-06-01", "2024-06-02", "2024-06-03", "2024-06-04", "2024-06-05", "2024-06-06", "2024-06-07", "2024-06-08", "2024-06-09", "2024-06-10", "2024-06-11", "2024-06-12", "2024-06-13", "2024-06-14", "2024-06-15", "2024-06-16", "2024-06-17", "2024-06-18", "2024-06-19", "2024-06-20", "2024-06-21", "2024-06-22", "2024-06-23", "2024-06-24", "2024-06-25", "2024-06-26", "2024-06-27", "2024-06-28", "2024-06-29", "2024-06-30", "2024-07-01", "2024-07-02", "2024-07-03", "2024-07-04", "2024-07-05", "2024-07-06", "2024-07-07", "2024-07-08", "2024-07-09", "2024-07-10", "2024-07-11", "2024-07-12", "2024-07-13", "2024-07-14", "2024-07-15", "2024-07-16", "2024-07-17", "2024-07-18", "2024-07-19", "2024-07-20", "2024-07-21", "2024-07-22", "2024-07-23", "2024-07-24", "2024-07-25", "2024-07-26", "2024-07-27", "2024-07-28", "2024-07-29", "2024-07-30", "2024-07-31", "2024-08-01", "2024-08-02", "2024-08-03", "2024-08-04", "2024-08-05", "2024-08-06", "2024-08-07", "2024-08-08", "2024-08-09", "2024-08-10", "2024-08-11", "2024-08-12", "2024-08-13", "2024-08-14", "2024-08-15", "2024-08-16", "2024-08-17", "2024-08-18", "2024-08-19", "2024-08-20", "2024-08-21", "2024-08-22", "2024-08-23", "2024-08-24", "2024-08-25", "2024-08-26", "2024-08-27", "2024-08-28", "2024-08-29", "2024-08-30", "2024-08-31", "2024-09-01", "2024-09-02", "2024-09-03", "2024-09-04", "2024-09-05", "2024-09-06", "2024-09-07", "2024-09-08", "2024-09-09", "2024-09-10", "2024-09-11", "2024-09-12", "2024-09-13", "2024-09-14", "2024-09-15", "2024-09-16", "2024-09-17", "2024-09-18", "2024-09-19", "2024-09-20", "2024-09-21", "2024-09-22", "2024-09-23", "2024-09-24", "2024-09-25", "2024-09-26", "2024-09-27", "2024-09-28", "2024-09-29", "2024-09-30", "2024-10-01", "2024-10-02", "2024-10-03", "2024-10-04", "2024-10-05", "2024-10-06", "2024-10-07", "2024-10-08", "2024-10-09", "2024-10-10", "2024-10-11", "2024-10-12", "2024-10-13", "2024-10-14", "2024-10-15", "2024-10-16", "2024-10-17", "2024-10-18", "2024-10-19", "2024-10-20", "2024-10-21", "2024-10-22", "2024-10-23", "2024-10-24", "2024-10-25", "2024-10-26", "2024-10-27", "2024-10-28", "2024-10-29", "2024-10-30", "2024-10-31", "2024-11-01", "2024-11-02", "2024-11-03", "2024-11-04", "2024-11-05", "2024-11-06", "2024-11-07", "2024-11-08", "2024-11-09", "2024-11-10", "2024-11-11", "2024-11-12", "2024-11-13", "2024-11-14", "2024-11-15", "2024-11-16", "2024-11-17", "2024-11-18", "2024-11-19", "2024-11-20", "2024-11-21", "2024-11-22", "2024-11-23", "2024-11-24", "2024-11-25", "2024-11-26", "2024-11-27", "2024-11-28", "2024-11-29", "2024-11-30", "2024-12-01", "2024-12-02", "2024-12-03", "2024-12-04", "2024-12-05", "2024-12-06", "2024-12-07", "2024-12-08", "2024-12-09", "2024-12-10", "2024-12-11", "2024-12-12", "2024-12-13", "2024-12-14", "2024-12-15", "2024-12-16", "2024-12-17", "2024-12-18", "2024-12-19", "2024-12-20", "2024-12-21", "2024-12-22", "2024-12-23", "2024-12-24", "2024-12-25", "2024-12-26", "2024-12-27", "2024-12-28", "2024-12-29", "2024-12-30", "2024-12-31", "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05", "2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09", "2025-01-10", "2025-01-11", "2025-01-12", "2025-01-13", "2025-01-14", "2025-01-15", "2025-01-16", "2025-01-17", "2025-01-18", "2025-01-19", "2025-01-20", "2025-01-21", "2025-01-22", "2025-01-23", "2025-01-24", "2025-01-25", "2025-01-26", "2025-01-27", "2025-01-28", "2025-01-29", "2025-01-30", "2025-01-31", "2025-02-01", "2025-02-02", "2025-02-03", "2025-02-04", "2025-02-05", "2025-02-06", "2025-02-07", "2025-02-08", "2025-02-09", "2025-02-10", "2025-02-11", "2025-02-12", "2025-02-13", "2025-02-14", "2025-02-15", "2025-02-16", "2025-02-17", "2025-02-18", "2025-02-19", "2025-02-20", "2025-02-21", "2025-02-22", "2025-02-23", "2025-02-24", "2025-02-25", "2025-02-26", "2025-02-27", "2025-02-28", "2025-03-01", "2025-03-02", "2025-03-03", "2025-03-04", "2025-03-05", "2025-03-06", "2025-03-07", "2025-03-08", "2025-03-09", "2025-03-10", "2025-03-11", "2025-03-12", "2025-03-13", "2025-03-14", "2025-03-15", "2025-03-16", "2025-03-17", "2025-03-18", "2025-03-19", "2025-03-20", "2025-03-21", "2025-03-22", "2025-03-23", "2025-03-24", "2025-03-25", "2025-03-26", "2025-03-27", "2025-03-28", "2025-03-29", "2025-03-30", "2025-03-31", "2025-04-01", "2025-04-02", "2025-04-03", "2025-04-04", "2025-04-05", "2025-04-06", "2025-04-07", "2025-04-08", "2025-04-09", "2025-04-10", "2025-04-11", "2025-04-12", "2025-04-13", "2025-04-14", "2025-04-15", "2025-04-16", "2025-04-17", "2025-04-18", "2025-04-19", "2025-04-20", "2025-04-21", "2025-04-22", "2025-04-23", "2025-04-24", "2025-04-25", "2025-04-26", "2025-04-27", "2025-04-28", "2025-04-29", "2025-04-30", "2025-05-01", "2025-05-02", "2025-05-03", "2025-05-04", "2025-05-05", "2025-05-06", "2025-05-07", "2025-05-08", "2025-05-09", "2025-05-10", "2025-05-11", "2025-05-12", "2025-05-13", "2025-05-14", "2025-05-15", "2025-05-16", "2025-05-17", "2025-05-18", "2025-05-19", "2025-05-20", "2025-05-21", "2025-05-22", "2025-05-23", "2025-05-24", "2025-05-25", "2025-05-26", "2025-05-27", "2025-05-28", "2025-05-29", "2025-05-30", "2025-05-31", "2025-06-01", "2025-06-02", "2025-06-03", "2025-06-04", "2025-06-05", "2025-06-06", "2025-06-07", "2025-06-08", "2025-06-09", "2025-06-10", "2025-06-11", "2025-06-12", "2025-06-13", "2025-06-14", "2025-06-15", "2025-06-16", "2025-06-17", "2025-06-18", "2025-06-19", "2025-06-20", "2025-06-21", "2025-06-22", "2025-06-23", "2025-06-24", "2025-06-25", "2025-06-26", "2025-06-27", "2025-06-28", "2025-06-29", "2025-06-30", "2025-07-01", "2025-07-02", "2025-07-03", "2025-07-04", "2025-07-05", "2025-07-06", "2025-07-07", "2025-07-08", "2025-07-09", "2025-07-10", "2025-07-11", "2025-07-12", "2025-07-13", "2025-07-14", "2025-07-15", "2025-07-16", "2025-07-17", "2025-07-18", "2025-07-19", "2025-07-20", "2025-07-21", "2025-07-22", "2025-07-23", "2025-07-24", "2025-07-25", "2025-07-26", "2025-07-27", "2025-07-28", "2025-07-29", "2025-07-30", "2025-07-31", "2025-08-01", "2025-08-02", "2025-08-03", "2025-08-04", "2025-08-05", "2025-08-06", "2025-08-07", "2025-08-08", "2025-08-09", "2025-08-10", "2025-08-11", "2025-08-12", "2025-08-13", "2025-08-14", "2025-08-15", "2025-08-16", "2025-08-17", "2025-08-18", "2025-08-19", "2025-08-20", "2025-08-21", "2025-08-22", "2025-08-23", "2025-08-24", "2025-08-25", "2025-08-26", "2025-08-27", "2025-08-28", "2025-08-29", "2025-08-30", "2025-08-31", "2025-09-01", "2025-09-02", "2025-09-03", "2025-09-04", "2025-09-05", "2025-09-06", "2025-09-07", "2025-09-08", "2025-09-09", "2025-09-10", "2025-09-11", "2025-09-12", "2025-09-13", "2025-09-14", "2025-09-15", "2025-09-16", "2025-09-17", "2025-09-18", "2025-09-19", "2025-09-20", "2025-09-21", "2025-09-22", "2025-09-23", "2025-09-24", "2025-09-25", "2025-09-26", "2025-09-27", "2025-09-28", "2025-09-29", "2025-09-30", "2025-10-01", "2025-10-02", "2025-10-03", "2025-10-04", "2025-10-05", "2025-10-06", "2025-10-07", "2025-10-08", "2025-10-09", "2025-10-10", "2025-10-11", "2025-10-12", "2025-10-13", "2025-10-14", "2025-10-15", "2025-10-16", "2025-10-17", "2025-10-18", "2025-10-19", "2025-10-20", "2025-10-21", "2025-10-22", "2025-10-23", "2025-10-24", "2025-10-25", "2025-10-26", "2025-10-27", "2025-10-28", "2025-10-29", "2025-10-30", "2025-10-31", "2025-11-01", "2025-11-02", "2025-11-03", "2025-11-04", "2025-11-05", "2025-11-06", "2025-11-07", "2025-11-08", "2025-11-09", "2025-11-10", "2025-11-11", "2025-11-12", "2025-11-13", "2025-11-14", "2025-11-15", "2025-11-16", "2025-11-17", "2025-11-18", "2025-11-19", "2025-11-20", "2025-11-21", "2025-11-22", "2025-11-23", "2025-11-24", "2025-11-25", "2025-11-26", "2025-11-27", "2025-11-28", "2025-11-29", "2025-11-30", "2025-12-01", "2025-12-02", "2025-12-03", "2025-12-04", "2025-12-05", "2025-12-06", "2025-12-07", "2025-12-08", "2025-12-09", "2025-12-10", "2025-12-11", "2025-12-12", "2025-12-13", "2025-12-14", "2025-12-15", "2025-12-16", "2025-12-17", "2025-12-18", "2025-12-19", "2025-12-20", "2025-12-21", "2025-12-22", "2025-12-23", "2025-12-24", "2025-12-25", "2025-12-26", "2025-12-27", "2025-12-28", "2025-12-29", "2025-12-30", "2025-12-31", "2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09", "2026-01-10", "2026-01-11", "2026-01-12", "2026-01-13", "2026-01-14", "2026-01-15", "2026-01-16", "2026-01-17", "2026-01-18", "2026-01-19", "2026-01-20", "2026-01-21", "2026-01-22", "2026-01-23", "2026-01-24", "2026-01-25", "2026-01-26", "2026-01-27", "2026-01-28", "2026-01-29", "2026-01-30", "2026-01-31", "2026-02-01", "2026-02-02", "2026-02-03", "2026-02-04", "2026-02-05", "2026-02-06", "2026-02-07", "2026-02-08", "2026-02-09", "2026-02-10", "2026-02-11", "2026-02-12", "2026-02-13", "2026-02-14", "2026-02-15", "2026-02-16", "2026-02-17", "2026-02-18", "2026-02-19", "2026-02-20", "2026-02-21", "2026-02-22", "2026-02-23", "2026-02-24", "2026-02-25", "2026-02-26", "2026-02-27", "2026-02-28", "2026-03-01", "2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05", "2026-03-06", "2026-03-07", "2026-03-08", "2026-03-09", "2026-03-10", "2026-03-11", "2026-03-12", "2026-03-13", "2026-03-14", "2026-03-15", "2026-03-16", "2026-03-17", "2026-03-18", "2026-03-19", "2026-03-20", "2026-03-21", "2026-03-22", "2026-03-23", "2026-03-24", "2026-03-25", "2026-03-26", "2026-03-27", "2026-03-28", "2026-03-29", "2026-03-30", "2026-03-31", "2026-04-01", "2026-04-02", "2026-04-03", "2026-04-04", "2026-04-05", "2026-04-06", "2026-04-07", "2026-04-08", "2026-04-09", "2026-04-10", "2026-04-11", "2026-04-12", "2026-04-13", "2026-04-14", "2026-04-15", "2026-04-16", "2026-04-17", "2026-04-18", "2026-04-19", "2026-04-20", "2026-04-21", "2026-04-22", "2026-04-23", "2026-04-24", "2026-04-25", "2026-04-26", "2026-04-27", "2026-04-28", "2026-04-29", "2026-04-30", "2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05", "2026-05-06", "2026-05-07", "2026-05-08", "2026-05-09", "2026-05-10", "2026-05-11", "2026-05-12", "2026-05-13", "2026-05-14", "2026-05-15", "2026-05-16", "2026-05-17", "2026-05-18", "2026-05-19", "2026-05-20", "2026-05-21", "2026-05-22", "2026-05-23", "2026-05-24", "2026-05-25", "2026-05-26", "2026-05-27", "2026-05-28", "2026-05-29", "2026-05-30", "2026-05-31", "2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05", "2026-06-06", "2026-06-07", "2026-06-08", "2026-06-09", "2026-06-10", "2026-06-11", "2026-06-12", "2026-06-13", "2026-06-14", "2026-06-15", "2026-06-16", "2026-06-17", "2026-06-18", "2026-06-19", "2026-06-20", "2026-06-21", "2026-06-22", "2026-06-23", "2026-06-24", "2026-06-25", "2026-06-26", "2026-06-27", "2026-06-28", "2026-06-29", "2026-06-30", "2026-07-01", "2026-07-02", "2026-07-03", "2026-07-04", "2026-07-05", "2026-07-06", "2026-07-07", "2026-07-08", "2026-07-09", "2026-07-10", "2026-07-11", "2026-07-12", "2026-07-13", "2026-07-14", "2026-07-15", "2026-07-16", "2026-07-17", "2026-07-18", "2026-07-19", "2026-07-20", "2026-07-21", "2026-07-22", "2026-07-23", "2026-07-24", "2026-07-25", "2026-07-26", "2026-07-27", "2026-07-28", "2026-07-29", "2026-07-30", "2026-07-31", "2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07", "2026-08-08", "2026-08-09", "2026-08-10", "2026-08-11", "2026-08-12", "2026-08-13", "2026-08-14", "2026-08-15", "2026-08-16", "2026-08-17", "2026-08-18", "2026-08-19", "2026-08-20", "2026-08-21", "2026-08-22", "2026-08-23", "2026-08-24", "2026-08-25", "2026-08-26", "2026-08-27", "2026-08-28", "2026-08-29", "2026-08-30", "2026-08-31", "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04", "2026-09-05", "2026-09-06", "2026-09-07", "2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11", "2026-09-12", "2026-09-13", "2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18", "2026-09-19", "2026-09-20", "2026-09-21", "2026-09-22", "2026-09-23", "2026-09-24", "2026-09-25", "2026-09-26", "2026-09-27", "2026-09-28", "2026-09-29", "2026-09-30", "2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08", "2026-10-09", "2026-10-10", "2026-10-11", "2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15", "2026-10-16", "2026-10-17", "2026-10-18", "2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22", "2026-10-23", "2026-10-24", "2026-10-25", "2026-10-26", "2026-10-27", "2026-10-28", "2026-10-29", "2026-10-30", "2026-10-31", "2026-11-01", "2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05", "2026-11-06", "2026-11-07", "2026-11-08", "2026-11-09", "2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13", "2026-11-14", "2026-11-15", "2026-11-16", "2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20", "2026-11-21", "2026-11-22", "2026-11-23", "2026-11-24", "2026-11-25", "2026-11-26", "2026-11-27", "2026-11-28", "2026-11-29", "2026-11-30", "2026-12-01", "2026-12-02", "2026-12-03", "2026-12-04", "2026-12-05", "2026-12-06", "2026-12-07", "2026-12-08", "2026-12-09", "2026-12-10", "2026-12-11", "2026-12-12", "2026-12-13", "2026-12-14", "2026-12-15", "2026-12-16", "2026-12-17", "2026-12-18", "2026-12-19", "2026-12-20", "2026-12-21", "2026-12-22", "2026-12-23", "2026-12-24", "2026-12-25", "2026-12-26", "2026-12-27", "2026-12-28", "2026-12-29", "2026-12-30", "2026-12-31", "2027-01-01", "2027-01-02", "2027-01-03", "2027-01-04", "2027-01-05", "2027-01-06", "2027-01-07", "2027-01-08", "2027-01-09", "2027-01-10", "2027-01-11", "2027-01-12", "2027-01-13", "2027-01-14", "2027-01-15", "2027-01-16", "2027-01-17", "2027-01-18", "2027-01-19", "2027-01-20", "2027-01-21", "2027-01-22", "2027-01-23", "2027-01-24", "2027-01-25", "2027-01-26", "2027-01-27", "2027-01-28", "2027-01-29", "2027-01-30", "2027-01-31", "2027-02-01", "2027-02-02", "2027-02-03", "2027-02-04", "2027-02-05", "2027-02-06", "2027-02-07", "2027-02-08", "2027-02-09", "2027-02-10", "2027-02-11", "2027-02-12", "2027-02-13", "2027-02-14", "2027-02-15", "2027-02-16", "2027-02-17", "2027-02-18", "2027-02-19", "2027-02-20", "2027-02-21", "2027-02-22", "2027-02-23", "2027-02-24", "2027-02-25", "2027-02-26", "2027-02-27", "2027-02-28", "2027-03-01", "2027-03-02", "2027-03-03", "2027-03-04", "2027-03-05", "2027-03-06", "2027-03-07", "2027-03-08", "2027-03-09", "2027-03-10", "2027-03-11", "2027-03-12", "2027-03-13", "2027-03-14", "2027-03-15", "2027-03-16", "2027-03-17", "2027-03-18", "2027-03-19", "2027-03-20", "2027-03-21", "2027-03-22", "2027-03-23", "2027-03-24", "2027-03-25", "2027-03-26", "2027-03-27", "2027-03-28", "2027-03-29", "2027-03-30", "2027-03-31", "2027-04-01", "2027-04-02", "2027-04-03", "2027-04-04", "2027-04-05", "2027-04-06", "2027-04-07", "2027-04-08", "2027-04-09", "2027-04-10", "2027-04-11", "2027-04-12", "2027-04-13", "2027-04-14", "2027-04-15", "2027-04-16", "2027-04-17", "2027-04-18", "2027-04-19", "2027-04-20", "2027-04-21", "2027-04-22", "2027-04-23", "2027-04-24", "2027-04-25", "2027-04-26", "2027-04-27", "2027-04-28", "2027-04-29", "2027-04-30", "2027-05-01", "2027-05-02", "2027-05-03", "2027-05-04", "2027-05-05", "2027-05-06", "2027-05-07", "2027-05-08", "2027-05-09", "2027-05-10", "2027-05-11", "2027-05-12", "2027-05-13", "2027-05-14", "2027-05-15", "2027-05-16", "2027-05-17", "2027-05-18", "2027-05-19", "2027-05-20", "2027-05-21", "2027-05-22", "2027-05-23", "2027-05-24", "2027-05-25", "2027-05-26", "2027-05-27", "2027-05-28", "2027-05-29", "2027-05-30", "2027-05-31", "2027-06-01", "2027-06-02", "2027-06-03", "2027-06-04", "2027-06-05", "2027-06-06", "2027-06-07", "2027-06-08", "2027-06-09", "2027-06-10", "2027-06-11", "2027-06-12", "2027-06-13", "2027-06-14", "2027-06-15", "2027-06-16", "2027-06-17", "2027-06-18", "2027-06-19", "2027-06-20", "2027-06-21", "2027-06-22", "2027-06-23", "2027-06-24", "2027-06-25", "2027-06-26", "2027-06-27", "2027-06-28", "2027-06-29", "2027-06-30", "2027-07-01", "2027-07-02", "2027-07-03", "2027-07-04", "2027-07-05", "2027-07-06", "2027-07-07", "2027-07-08", "2027-07-09", "2027-07-10", "2027-07-11", "2027-07-12", "2027-07-13", "2027-07-14", "2027-07-15", "2027-07-16", "2027-07-17", "2027-07-18", "2027-07-19", "2027-07-20", "2027-07-21", "2027-07-22", "2027-07-23", "2027-07-24", "2027-07-25", "2027-07-26", "2027-07-27", "2027-07-28", "2027-07-29", "2027-07-30", "2027-07-31", "2027-08-01", "2027-08-02", "2027-08-03", "2027-08-04", "2027-08-05", "2027-08-06", "2027-08-07", "2027-08-08", "2027-08-09", "2027-08-10", "2027-08-11", "2027-08-12", "2027-08-13", "2027-08-14", "2027-08-15", "2027-08-16", "2027-08-17", "2027-08-18", "2027-08-19", "2027-08-20", "2027-08-21", "2027-08-22", "2027-08-23", "2027-08-24", "2027-08-25", "2027-08-26", "2027-08-27", "2027-08-28", "2027-08-29", "2027-08-30", "2027-08-31", "2027-09-01", "2027-09-02", "2027-09-03", "2027-09-04", "2027-09-05", "2027-09-06", "2027-09-07", "2027-09-08", "2027-09-09", "2027-09-10", "2027-09-11", "2027-09-12", "2027-09-13", "2027-09-14", "2027-09-15", "2027-09-16", "2027-09-17", "2027-09-18", "2027-09-19", "2027-09-20", "2027-09-21", "2027-09-22", "2027-09-23", "2027-09-24", "2027-09-25", "2027-09-26", "2027-09-27", "2027-09-28", "2027-09-29", "2027-09-30", "2027-10-01", "2027-10-02", "2027-10-03", "2027-10-04", "2027-10-05", "2027-10-06", "2027-10-07", "2027-10-08", "2027-10-09", "2027-10-10", "2027-10-11", "2027-10-12", "2027-10-13", "2027-10-14", "2027-10-15", "2027-10-16", "2027-10-17", "2027-10-18", "2027-10-19", "2027-10-20", "2027-10-21", "2027-10-22", "2027-10-23", "2027-10-24", "2027-10-25", "2027-10-26", "2027-10-27", "2027-10-28", "2027-10-29", "2027-10-30", "2027-10-31", "2027-11-01", "2027-11-02", "2027-11-03", "2027-11-04", "2027-11-05", "2027-11-06", "2027-11-07", "2027-11-08", "2027-11-09", "2027-11-10", "2027-11-11", "2027-11-12", "2027-11-13", "2027-11-14", "2027-11-15", "2027-11-16", "2027-11-17", "2027-11-18", "2027-11-19", "2027-11-20", "2027-11-21", "2027-11-22", "2027-11-23", "2027-11-24", "2027-11-25", "2027-11-26", "2027-11-27", "2027-11-28", "2027-11-29", "2027-11-30", "2027-12-01", "2027-12-02", "2027-12-03", "2027-12-04", "2027-12-05", "2027-12-06", "2027-12-07", "2027-12-08", "2027-12-09", "2027-12-10", "2027-12-11", "2027-12-12", "2027-12-13", "2027-12-14", "2027-12-15", "2027-12-16", "2027-12-17", "2027-12-18", "2027-12-19", "2027-12-20", "2027-12-21", "2027-12-22", "2027-12-23", "2027-12-24", "2027-12-25", "2027-12-26", "2027-12-27", "2027-12-28", "2027-12-29", "2027-12-30", "2027-12-31", "2028-01-01", "2028-01-02", "2028-01-03", "2028-01-04", "2028-01-05", "2028-01-06", "2028-01-07", "2028-01-08", "2028-01-09", "2028-01-10", "2028-01-11", "2028-01-12", "2028-01-13", "2028-01-14", "2028-01-15", "2028-01-16", "2028-01-17", "2028-01-18", "2028-01-19", "2028-01-20", "2028-01-21", "2028-01-22", "2028-01-23", "2028-01-24", "2028-01-25", "2028-01-26", "2028-01-27", "2028-01-28", "2028-01-29", "2028-01-30", "2028-01-31", "2028-02-01", "2028-02-02", "2028-02-03", "2028-02-04", "2028-02-05", "2028-02-06", "2028-02-07", "2028-02-08", "2028-02-09", "2028-02-10", "2028-02-11", "2028-02-12", "2028-02-13", "2028-02-14", "2028-02-15", "2028-02-16", "2028-02-17", "2028-02-18", "2028-02-19", "2028-02-20", "2028-02-21", "2028-02-22", "2028-02-23", "2028-02-24", "2028-02-25", "2028-02-26", "2028-02-27", "2028-02-28", "2028-02-29", "2028-03-01", "2028-03-02", "2028-03-03", "2028-03-04", "2028-03-05", "2028-03-06", "2028-03-07", "2028-03-08", "2028-03-09", "2028-03-10", "2028-03-11", "2028-03-12", "2028-03-13", "2028-03-14", "2028-03-15", "2028-03-16", "2028-03-17", "2028-03-18", "2028-03-19", "2028-03-20", "2028-03-21", "2028-03-22", "2028-03-23", "2028-03-24", "2028-03-25", "2028-03-26", "2028-03-27", "2028-03-28", "2028-03-29", "2028-03-30", "2028-03-31", "2028-04-01", "2028-04-02", "2028-04-03", "2028-04-04", "2028-04-05", "2028-04-06", "2028-04-07", "2028-04-08", "2028-04-09", "2028-04-10", "2028-04-11", "2028-04-12", "2028-04-13", "2028-04-14", "2028-04-15", "2028-04-16", "2028-04-17", "2028-04-18", "2028-04-19", "2028-04-20", "2028-04-21", "2028-04-22", "2028-04-23", "2028-04-24", "2028-04-25", "2028-04-26", "2028-04-27", "2028-04-28", "2028-04-29", "2028-04-30", "2028-05-01", "2028-05-02", "2028-05-03", "2028-05-04", "2028-05-05", "2028-05-06", "2028-05-07", "2028-05-08", "2028-05-09", "2028-05-10", "2028-05-11", "2028-05-12", "2028-05-13", "2028-05-14", "2028-05-15", "2028-05-16", "2028-05-17", "2028-05-18", "2028-05-19", "2028-05-20", "2028-05-21", "2028-05-22", "2028-05-23", "2028-05-24", "2028-05-25", "2028-05-26", "2028-05-27", "2028-05-28", "2028-05-29", "2028-05-30", "2028-05-31", "2028-06-01", "2028-06-02", "2028-06-03", "2028-06-04", "2028-06-05", "2028-06-06", "2028-06-07", "2028-06-08", "2028-06-09", "2028-06-10", "2028-06-11", "2028-06-12", "2028-06-13", "2028-06-14", "2028-06-15", "2028-06-16", "2028-06-17", "2028-06-18", "2028-06-19", "2028-06-20", "2028-06-21", "2028-06-22", "2028-06-23", "2028-06-24", "2028-06-25", "2028-06-26", "2028-06-27", "2028-06-28", "2028-06-29", "2028-06-30", "2028-07-01", "2028-07-02", "2028-07-03", "2028-07-04", "2028-07-05", "2028-07-06", "2028-07-07", "2028-07-08", "2028-07-09", "2028-07-10", "2028-07-11", "2028-07-12", "2028-07-13", "2028-07-14", "2028-07-15", "2028-07-16", "2028-07-17", "2028-07-18", "2028-07-19", "2028-07-20", "2028-07-21", "2028-07-22", "2028-07-23", "2028-07-24", "2028-07-25", "2028-07-26", "2028-07-27", "2028-07-28", "2028-07-29", "2028-07-30", "2028-07-31", "2028-08-01", "2028-08-02", "2028-08-03", "2028-08-04", "2028-08-05", "2028-08-06", "2028-08-07", "2028-08-08", "2028-08-09", "2028-08-10", "2028-08-11", "2028-08-12", "2028-08-13", "2028-08-14", "2028-08-15", "2028-08-16", "2028-08-17", "2028-08-18", "2028-08-19", "2028-08-20", "2028-08-21", "2028-08-22", "2028-08-23", "2028-08-24", "2028-08-25", "2028-08-26", "2028-08-27", "2028-08-28", "2028-08-29", "2028-08-30", "2028-08-31", "2028-09-01", "2028-09-02", "2028-09-03", "2028-09-04", "2028-09-05", "2028-09-06", "2028-09-07", "2028-09-08", "2028-09-09", "2028-09-10", "2028-09-11", "2028-09-12", "2028-09-13", "2028-09-14", "2028-09-15", "2028-09-16", "2028-09-17", "2028-09-18", "2028-09-19", "2028-09-20", "2028-09-21", "2028-09-22", "2028-09-23", "2028-09-24", "2028-09-25", "2028-09-26", "2028-09-27", "2028-09-28", "2028-09-29", "2028-09-30", "2028-10-01", "2028-10-02", "2028-10-03", "2028-10-04", "2028-10-05", "2028-10-06", "2028-10-07", "2028-10-08", "2028-10-09", "2028-10-10", "2028-10-11", "2028-10-12", "2028-10-13", "2028-10-14", "2028-10-15", "2028-10-16", "2028-10-17", "2028-10-18", "2028-10-19", "2028-10-20", "2028-10-21", "2028-10-22", "2028-10-23", "2028-10-24", "2028-10-25", "2028-10-26", "2028-10-27", "2028-10-28", "2028-10-29", "2028-10-30", "2028-10-31", "2028-11-01", "2028-11-02", "2028-11-03", "2028-11-04", "2028-11-05", "2028-11-06", "2028-11-07", "2028-11-08", "2028-11-09", "2028-11-10", "2028-11-11", "2028-11-12", "2028-11-13", "2028-11-14", "2028-11-15", "2028-11-16", "2028-11-17", "2028-11-18", "2028-11-19", "2028-11-20", "2028-11-21", "2028-11-22", "2028-11-23", "2028-11-24", "2028-11-25", "2028-11-26", "2028-11-27", "2028-11-28", "2028-11-29", "2028-11-30", "2028-12-01", "2028-12-02", "2028-12-03", "2028-12-04", "2028-12-05", "2028-12-06", "2028-12-07", "2028-12-08", "2028-12-09", "2028-12-10", "2028-12-11", "2028-12-12", "2028-12-13", "2028-12-14", "2028-12-15", "2028-12-16", "2028-12-17", "2028-12-18", "2028-12-19", "2028-12-20", "2028-12-21", "2028-12-22", "2028-12-23", "2028-12-24", "2028-12-25", "2028-12-26", "2028-12-27", "2028-12-28", "2028-12-29", "2028-12-30", "2028-12-31", "2029-01-01", "2029-01-02", "2029-01-03", "2029-01-04", "2029-01-05", "2029-01-06", "2029-01-07", "2029-01-08", "2029-01-09", "2029-01-10", "2029-01-11", "2029-01-12", "2029-01-13", "2029-01-14", "2029-01-15", "2029-01-16", "2029-01-17", "2029-01-18", "2029-01-19", "2029-01-20", "2029-01-21", "2029-01-22", "2029-01-23", "2029-01-24", "2029-01-25", "2029-01-26", "2029-01-27", "2029-01-28", "2029-01-29", "2029-01-30", "2029-01-31", "2029-02-01", "2029-02-02", "2029-02-03", "2029-02-04", "2029-02-05", "2029-02-06", "2029-02-07", "2029-02-08", "2029-02-09", "2029-02-10", "2029-02-11", "2029-02-12", "2029-02-13", "2029-02-14", "2029-02-15", "2029-02-16", "2029-02-17", "2029-02-18", "2029-02-19", "2029-02-20", "2029-02-21", "2029-02-22", "2029-02-23", "2029-02-24", "2029-02-25", "2029-02-26", "2029-02-27", "2029-02-28", "2029-03-01", "2029-03-02", "2029-03-03", "2029-03-04", "2029-03-05", "2029-03-06", "2029-03-07", "2029-03-08", "2029-03-09", "2029-03-10", "2029-03-11", "2029-03-12", "2029-03-13", "2029-03-14", "2029-03-15", "2029-03-16", "2029-03-17", "2029-03-18", "2029-03-19", "2029-03-20", "2029-03-21", "2029-03-22", "2029-03-23", "2029-03-24", "2029-03-25", "2029-03-26", "2029-03-27", "2029-03-28", "2029-03-29", "2029-03-30", "2029-03-31", "2029-04-01", "2029-04-02", "2029-04-03", "2029-04-04", "2029-04-05", "2029-04-06", "2029-04-07", "2029-04-08", "2029-04-09", "2029-04-10", "2029-04-11", "2029-04-12", "2029-04-13", "2029-04-14", "2029-04-15", "2029-04-16", "2029-04-17", "2029-04-18", "2029-04-19", "2029-04-20", "2029-04-21", "2029-04-22", "2029-04-23", "2029-04-24", "2029-04-25", "2029-04-26", "2029-04-27", "2029-04-28", "2029-04-29", "2029-04-30", "2029-05-01", "2029-05-02", "2029-05-03", "2029-05-04", "2029-05-05", "2029-05-06", "2029-05-07", "2029-05-08", "2029-05-09", "2029-05-10", "2029-05-11", "2029-05-12", "2029-05-13", "2029-05-14", "2029-05-15", "2029-05-16", "2029-05-17", "2029-05-18", "2029-05-19", "2029-05-20", "2029-05-21", "2029-05-22", "2029-05-23", "2029-05-24", "2029-05-25", "2029-05-26", "2029-05-27", "2029-05-28", "2029-05-29", "2029-05-30", "2029-05-31", "2029-06-01", "2029-06-02", "2029-06-03", "2029-06-04", "2029-06-05", "2029-06-06", "2029-06-07", "2029-06-08", "2029-06-09", "2029-06-10", "2029-06-11", "2029-06-12", "2029-06-13", "2029-06-14", "2029-06-15", "2029-06-16", "2029-06-17", "2029-06-18", "2029-06-19", "2029-06-20", "2029-06-21", "2029-06-22", "2029-06-23", "2029-06-24", "2029-06-25", "2029-06-26", "2029-06-27", "2029-06-28", "2029-06-29", "2029-06-30", "2029-07-01", "2029-07-02", "2029-07-03", "2029-07-04", "2029-07-05", "2029-07-06", "2029-07-07", "2029-07-08", "2029-07-09", "2029-07-10", "2029-07-11", "2029-07-12", "2029-07-13", "2029-07-14", "2029-07-15", "2029-07-16", "2029-07-17", "2029-07-18", "2029-07-19", "2029-07-20", "2029-07-21", "2029-07-22", "2029-07-23", "2029-07-24", "2029-07-25", "2029-07-26", "2029-07-27", "2029-07-28", "2029-07-29", "2029-07-30", "2029-07-31", "2029-08-01", "2029-08-02", "2029-08-03", "2029-08-04", "2029-08-05", "2029-08-06", "2029-08-07", "2029-08-08", "2029-08-09", "2029-08-10", "2029-08-11", "2029-08-12", "2029-08-13", "2029-08-14", "2029-08-15", "2029-08-16", "2029-08-17", "2029-08-18", "2029-08-19", "2029-08-20", "2029-08-21", "2029-08-22", "2029-08-23", "2029-08-24", "2029-08-25", "2029-08-26", "2029-08-27", "2029-08-28", "2029-08-29", "2029-08-30", "2029-08-31", "2029-09-01", "2029-09-02", "2029-09-03", "2029-09-04", "2029-09-05", "2029-09-06", "2029-09-07", "2029-09-08", "2029-09-09", "2029-09-10", "2029-09-11", "2029-09-12", "2029-09-13", "2029-09-14", "2029-09-15", "2029-09-16", "2029-09-17", "2029-09-18", "2029-09-19", "2029-09-20", "2029-09-21", "2029-09-22", "2029-09-23", "2029-09-24", "2029-09-25", "2029-09-26", "2029-09-27", "2029-09-28", "2029-09-29", "2029-09-30", "2029-10-01", "2029-10-02", "2029-10-03", "2029-10-04", "2029-10-05", "2029-10-06", "2029-10-07", "2029-10-08", "2029-10-09", "2029-10-10", "2029-10-11", "2029-10-12", "2029-10-13", "2029-10-14", "2029-10-15", "2029-10-16", "2029-10-17", "2029-10-18", "2029-10-19", "2029-10-20", "2029-10-21", "2029-10-22", "2029-10-23", "2029-10-24", "2029-10-25", "2029-10-26", "2029-10-27", "2029-10-28", "2029-10-29", "2029-10-30", "2029-10-31", "2029-11-01", "2029-11-02", "2029-11-03", "2029-11-04", "2029-11-05", "2029-11-06", "2029-11-07", "2029-11-08", "2029-11-09", "2029-11-10", "2029-11-11", "2029-11-12", "2029-11-13", "2029-11-14", "2029-11-15", "2029-11-16", "2029-11-17", "2029-11-18", "2029-11-19", "2029-11-20", "2029-11-21", "2029-11-22", "2029-11-23", "2029-11-24", "2029-11-25", "2029-11-26", "2029-11-27", "2029-11-28", "2029-11-29", "2029-11-30", "2029-12-01", "2029-12-02", "2029-12-03", "2029-12-04", "2029-12-05", "2029-12-06", "2029-12-07", "2029-12-08", "2029-12-09", "2029-12-10", "2029-12-11", "2029-12-12", "2029-12-13", "2029-12-14", "2029-12-15", "2029-12-16", "2029-12-17", "2029-12-18", "2029-12-19", "2029-12-20", "2029-12-21", "2029-12-22", "2029-12-23", "2029-12-24", "2029-12-25", "2029-12-26", "2029-12-27", "2029-12-28", "2029-12-29", "2029-12-30", "2029-12-31", "2030-01-01", "2030-01-02", "2030-01-03", "2030-01-04", "2030-01-05", "2030-01-06", "2030-01-07", "2030-01-08", "2030-01-09", "2030-01-10", "2030-01-11", "2030-01-12", "2030-01-13", "2030-01-14", "2030-01-15", "2030-01-16", "2030-01-17", "2030-01-18", "2030-01-19", "2030-01-20", "2030-01-21", "2030-01-22", "2030-01-23", "2030-01-24", "2030-01-25", "2030-01-26", "2030-01-27", "2030-01-28", "2030-01-29", "2030-01-30", "2030-01-31", "2030-02-01", "2030-02-02", "2030-02-03", "2030-02-04", "2030-02-05", "2030-02-06", "2030-02-07", "2030-02-08", "2030-02-09", "2030-02-10", "2030-02-11", "2030-02-12", "2030-02-13", "2030-02-14", "2030-02-15", "2030-02-16", "2030-02-17", "2030-02-18", "2030-02-19", "2030-02-20", "2030-02-21", "2030-02-22", "2030-02-23", "2030-02-24", "2030-02-25", "2030-02-26", "2030-02-27", "2030-02-28", "2030-03-01", "2030-03-02", "2030-03-03", "2030-03-04", "2030-03-05", "2030-03-06", "2030-03-07", "2030-03-08", "2030-03-09", "2030-03-10", "2030-03-11", "2030-03-12", "2030-03-13", "2030-03-14", "2030-03-15", "2030-03-16", "2030-03-17", "2030-03-18", "2030-03-19", "2030-03-20", "2030-03-21", "2030-03-22", "2030-03-23", "2030-03-24", "2030-03-25", "2030-03-26", "2030-03-27", "2030-03-28", "2030-03-29", "2030-03-30", "2030-03-31", "2030-04-01", "2030-04-02", "2030-04-03", "2030-04-04", "2030-04-05", "2030-04-06", "2030-04-07", "2030-04-08", "2030-04-09", "2030-04-10", "2030-04-11", "2030-04-12", "2030-04-13", "2030-04-14", "2030-04-15", "2030-04-16", "2030-04-17", "2030-04-18", "2030-04-19", "2030-04-20", "2030-04-21", "2030-04-22", "2030-04-23", "2030-04-24", "2030-04-25", "2030-04-26", "2030-04-27", "2030-04-28", "2030-04-29", "2030-04-30", "2030-05-01", "2030-05-02", "2030-05-03", "2030-05-04", "2030-05-05", "2030-05-06", "2030-05-07", "2030-05-08", "2030-05-09", "2030-05-10", "2030-05-11", "2030-05-12", "2030-05-13", "2030-05-14", "2030-05-15", "2030-05-16", "2030-05-17", "2030-05-18", "2030-05-19", "2030-05-20", "2030-05-21", "2030-05-22", "2030-05-23", "2030-05-24", "2030-05-25", "2030-05-26", "2030-05-27", "2030-05-28", "2030-05-29", "2030-05-30", "2030-05-31", "2030-06-01", "2030-06-02", "2030-06-03", "2030-06-04", "2030-06-05", "2030-06-06", "2030-06-07", "2030-06-08", "2030-06-09", "2030-06-10", "2030-06-11", "2030-06-12", "2030-06-13", "2030-06-14", "2030-06-15", "2030-06-16", "2030-06-17", "2030-06-18", "2030-06-19", "2030-06-20", "2030-06-21", "2030-06-22", "2030-06-23", "2030-06-24", "2030-06-25", "2030-06-26", "2030-06-27", "2030-06-28", "2030-06-29", "2030-06-30", "2030-07-01", "2030-07-02", "2030-07-03", "2030-07-04", "2030-07-05", "2030-07-06", "2030-07-07", "2030-07-08", "2030-07-09", "2030-07-10", "2030-07-11", "2030-07-12", "2030-07-13", "2030-07-14", "2030-07-15", "2030-07-16", "2030-07-17", "2030-07-18", "2030-07-19", "2030-07-20", "2030-07-21", "2030-07-22", "2030-07-23", "2030-07-24", "2030-07-25", "2030-07-26", "2030-07-27", "2030-07-28", "2030-07-29", "2030-07-30", "2030-07-31", "2030-08-01", "2030-08-02", "2030-08-03", "2030-08-04", "2030-08-05", "2030-08-06", "2030-08-07", "2030-08-08", "2030-08-09", "2030-08-10", "2030-08-11", "2030-08-12", "2030-08-13", "2030-08-14", "2030-08-15", "2030-08-16", "2030-08-17", "2030-08-18", "2030-08-19", "2030-08-20", "2030-08-21", "2030-08-22", "2030-08-23", "2030-08-24", "2030-08-25", "2030-08-26", "2030-08-27", "2030-08-28", "2030-08-29", "2030-08-30", "2030-08-31", "2030-09-01", "2030-09-02", "2030-09-03", "2030-09-04", "2030-09-05", "2030-09-06", "2030-09-07", "2030-09-08", "2030-09-09", "2030-09-10", "2030-09-11", "2030-09-12", "2030-09-13", "2030-09-14", "2030-09-15", "2030-09-16", "2030-09-17", "2030-09-18", "2030-09-19", "2030-09-20", "2030-09-21", "2030-09-22", "2030-09-23", "2030-09-24", "2030-09-25", "2030-09-26", "2030-09-27", "2030-09-28", "2030-09-29", "2030-09-30", "2030-10-01", "2030-10-02", "2030-10-03", "2030-10-04", "2030-10-05", "2030-10-06", "2030-10-07", "2030-10-08", "2030-10-09", "2030-10-10", "2030-10-11", "2030-10-12", "2030-10-13", "2030-10-14", "2030-10-15", "2030-10-16", "2030-10-17", "2030-10-18", "2030-10-19", "2030-10-20", "2030-10-21", "2030-10-22", "2030-10-23", "2030-10-24", "2030-10-25", "2030-10-26", "2030-10-27", "2030-10-28", "2030-10-29", "2030-10-30", "2030-10-31", "2030-11-01", "2030-11-02", "2030-11-03", "2030-11-04", "2030-11-05", "2030-11-06", "2030-11-07", "2030-11-08", "2030-11-09", "2030-11-10", "2030-11-11", "2030-11-12", "2030-11-13", "2030-11-14", "2030-11-15", "2030-11-16", "2030-11-17", "2030-11-18", "2030-11-19", "2030-11-20", "2030-11-21", "2030-11-22", "2030-11-23", "2030-11-24", "2030-11-25", "2030-11-26", "2030-11-27", "2030-11-28", "2030-11-29", "2030-11-30", "2030-12-01", "2030-12-02", "2030-12-03", "2030-12-04", "2030-12-05", "2030-12-06", "2030-12-07", "2030-12-08", "2030-12-09", "2030-12-10", "2030-12-11", "2030-12-12", "2030-12-13", "2030-12-14", "2030-12-15", "2030-12-16", "2030-12-17", "2030-12-18", "2030-12-19", "2030-12-20", "2030-12-21", "2030-12-22", "2030-12-23", "2030-12-24", "2030-12-25", "2030-12-26", "2030-12-27", "2030-12-28", "2030-12-29", "2030-12-30", "2030-12-31", "2031-01-01", "2031-01-02", "2031-01-03", "2031-01-04", "2031-01-05", "2031-01-06", "2031-01-07", "2031-01-08", "2031-01-09", "2031-01-10", "2031-01-11", "2031-01-12", "2031-01-13", "2031-01-14", "2031-01-15", "2031-01-16", "2031-01-17", "2031-01-18", "2031-01-19", "2031-01-20", "2031-01-21", "2031-01-22", "2031-01-23", "2031-01-24", "2031-01-25", "2031-01-26", "2031-01-27", "2031-01-28", "2031-01-29", "2031-01-30", "2031-01-31", "2031-02-01", "2031-02-02", "2031-02-03", "2031-02-04", "2031-02-05", "2031-02-06", "2031-02-07", "2031-02-08", "2031-02-09", "2031-02-10", "2031-02-11", "2031-02-12", "2031-02-13", "2031-02-14", "2031-02-15", "2031-02-16", "2031-02-17", "2031-02-18", "2031-02-19", "2031-02-20", "2031-02-21", "2031-02-22", "2031-02-23", "2031-02-24", "2031-02-25", "2031-02-26", "2031-02-27", "2031-02-28", "2031-03-01", "2031-03-02", "2031-03-03", "2031-03-04", "2031-03-05", "2031-03-06", "2031-03-07", "2031-03-08", "2031-03-09", "2031-03-10", "2031-03-11", "2031-03-12", "2031-03-13", "2031-03-14", "2031-03-15", "2031-03-16", "2031-03-17", "2031-03-18", "2031-03-19", "2031-03-20", "2031-03-21", "2031-03-22", "2031-03-23", "2031-03-24", "2031-03-25", "2031-03-26", "2031-03-27", "2031-03-28", "2031-03-29", "2031-03-30", "2031-03-31", "2031-04-01", "2031-04-02", "2031-04-03", "2031-04-04", "2031-04-05", "2031-04-06", "2031-04-07", "2031-04-08", "2031-04-09", "2031-04-10", "2031-04-11", "2031-04-12", "2031-04-13", "2031-04-14", "2031-04-15", "2031-04-16", "2031-04-17", "2031-04-18", "2031-04-19", "2031-04-20", "2031-04-21", "2031-04-22", "2031-04-23", "2031-04-24", "2031-04-25", "2031-04-26", "2031-04-27", "2031-04-28", "2031-04-29", "2031-04-30", "2031-05-01", "2031-05-02", "2031-05-03", "2031-05-04", "2031-05-05", "2031-05-06", "2031-05-07", "2031-05-08", "2031-05-09", "2031-05-10", "2031-05-11", "2031-05-12", "2031-05-13", "2031-05-14", "2031-05-15", "2031-05-16", "2031-05-17", "2031-05-18", "2031-05-19", "2031-05-20", "2031-05-21", "2031-05-22", "2031-05-23", "2031-05-24", "2031-05-25", "2031-05-26", "2031-05-27", "2031-05-28", "2031-05-29", "2031-05-30", "2031-05-31", "2031-06-01", "2031-06-02", "2031-06-03", "2031-06-04", "2031-06-05", "2031-06-06", "2031-06-07", "2031-06-08", "2031-06-09", "2031-06-10", "2031-06-11", "2031-06-12", "2031-06-13", "2031-06-14", "2031-06-15", "2031-06-16", "2031-06-17", "2031-06-18", "2031-06-19", "2031-06-20", "2031-06-21", "2031-06-22", "2031-06-23", "2031-06-24", "2031-06-25", "2031-06-26", "2031-06-27", "2031-06-28", "2031-06-29", "2031-06-30", "2031-07-01", "2031-07-02", "2031-07-03", "2031-07-04", "2031-07-05", "2031-07-06", "2031-07-07", "2031-07-08", "2031-07-09", "2031-07-10", "2031-07-11", "2031-07-12", "2031-07-13", "2031-07-14", "2031-07-15", "2031-07-16", "2031-07-17", "2031-07-18", "2031-07-19", "2031-07-20", "2031-07-21", "2031-07-22", "2031-07-23", "2031-07-24", "2031-07-25", "2031-07-26", "2031-07-27", "2031-07-28", "2031-07-29", "2031-07-30", "2031-07-31", "2031-08-01", "2031-08-02", "2031-08-03", "2031-08-04", "2031-08-05", "2031-08-06", "2031-08-07", "2031-08-08", "2031-08-09", "2031-08-10", "2031-08-11", "2031-08-12", "2031-08-13", "2031-08-14", "2031-08-15", "2031-08-16", "2031-08-17", "2031-08-18", "2031-08-19", "2031-08-20", "2031-08-21", "2031-08-22", "2031-08-23", "2031-08-24", "2031-08-25", "2031-08-26", "2031-08-27", "2031-08-28", "2031-08-29", "2031-08-30", "2031-08-31", "2031-09-01", "2031-09-02", "2031-09-03", "2031-09-04", "2031-09-05", "2031-09-06", "2031-09-07", "2031-09-08", "2031-09-09", "2031-09-10", "2031-09-11", "2031-09-12", "2031-09-13", "2031-09-14", "2031-09-15", "2031-09-16", "2031-09-17", "2031-09-18", "2031-09-19", "2031-09-20", "2031-09-21", "2031-09-22", "2031-09-23", "2031-09-24", "2031-09-25", "2031-09-26", "2031-09-27", "2031-09-28", "2031-09-29", "2031-09-30", "2031-10-01", "2031-10-02", "2031-10-03", "2031-10-04", "2031-10-05", "2031-10-06", "2031-10-07", "2031-10-08", "2031-10-09", "2031-10-10", "2031-10-11", "2031-10-12", "2031-10-13", "2031-10-14", "2031-10-15", "2031-10-16", "2031-10-17", "2031-10-18", "2031-10-19", "2031-10-20", "2031-10-21", "2031-10-22", "2031-10-23", "2031-10-24", "2031-10-25", "2031-10-26", "2031-10-27", "2031-10-28", "2031-10-29", "2031-10-30", "2031-10-31", "2031-11-01", "2031-11-02", "2031-11-03", "2031-11-04", "2031-11-05", "2031-11-06", "2031-11-07", "2031-11-08", "2031-11-09", "2031-11-10", "2031-11-11", "2031-11-12", "2031-11-13", "2031-11-14", "2031-11-15", "2031-11-16", "2031-11-17", "2031-11-18", "2031-11-19", "2031-11-20", "2031-11-21", "2031-11-22", "2031-11-23", "2031-11-24", "2031-11-25", "2031-11-26", "2031-11-27", "2031-11-28", "2031-11-29", "2031-11-30", "2031-12-01", "2031-12-02", "2031-12-03", "2031-12-04", "2031-12-05", "2031-12-06", "2031-12-07", "2031-12-08", "2031-12-09", "2031-12-10", "2031-12-11", "2031-12-12", "2031-12-13", "2031-12-14", "2031-12-15", "2031-12-16", "2031-12-17", "2031-12-18", "2031-12-19", "2031-12-20", "2031-12-21", "2031-12-22", "2031-12-23", "2031-12-24", "2031-12-25", "2031-12-26", "2031-12-27", "2031-12-28", "2031-12-29", "2031-12-30", "2031-12-31", "2032-01-01", "2032-01-02", "2032-01-03", "2032-01-04", "2032-01-05", "2032-01-06", "2032-01-07", "2032-01-08", "2032-01-09", "2032-01-10", "2032-01-11", "2032-01-12", "2032-01-13", "2032-01-14", "2032-01-15", "2032-01-16", "2032-01-17", "2032-01-18", "2032-01-19", "2032-01-20", "2032-01-21", "2032-01-22", "2032-01-23", "2032-01-24", "2032-01-25", "2032-01-26", "2032-01-27", "2032-01-28", "2032-01-29", "2032-01-30", "2032-01-31", "2032-02-01", "2032-02-02", "2032-02-03", "2032-02-04", "2032-02-05", "2032-02-06", "2032-02-07", "2032-02-08", "2032-02-09", "2032-02-10", "2032-02-11", "2032-02-12", "2032-02-13", "2032-02-14", "2032-02-15", "2032-02-16", "2032-02-17", "2032-02-18", "2032-02-19"], "tickvals": [0, 42, 87, 130, 171, 214, 256, 298, 346, 390, 434, 481, 523, 564, 605, 647, 690, 731, 773, 813, 853, 897, 942, 985, 1026, 1068, 1113, 1155, 1195, 1235, 1277, 1321, 1362, 1404, 1446, 1487, 1528, 1569, 1613, 1654, 1695, 1736, 1776, 1816, 1857, 1897, 1938, 1978, 2019, 2059, 2099, 2139, 2179, 2219, 2259, 2299, 2339, 2380, 2420, 2461, 2501, 2542, 2582, 2622, 2662, 2702, 2742, 2782, 2822, 2862, 2902, 2943, 2983, 3023, 3063, 3103, 3143, 3183, 3223, 3263, 3303, 3343, 3383, 3423, 3463, 3503, 3543, 3583, 3623, 3663, 3703, 3743, 3783, 3823, 3863, 3903, 3943, 3983, 4023, 4063, 4103, 4143, 4183, 4223, 4263, 4303, 4343], "title": {"text": "day_offset"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "workload"}}},
                        {"responsive": true}
                    ).then(function(){
                            
var gd = document.getElementById('f60bc0f6-d1d8-4444-9d52-b798310f4069');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
                        {% raw %}
if (notebookContainer) {{ x.observe(notebookContainer, {childList: true});}}
{% endraw %}
// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
{% raw %}
 if (outputEl) {{x.observe(outputEl, {childList: true});}}
{% endraw %}

                        })
                };
                });
            </script>
        </div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">

<div class="output_wrapper">
<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">
<pre>[[0.09857098]] [-11.25573084]
-2.044872940246718 43.194449250712374
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Linear-Regression-Model-Coeffs-and-Error:">Linear Regression Model Coeffs and Error:<a class="anchor-link" href="#Linear-Regression-Model-Coeffs-and-Error:">&#182;</a></h2><p>/The sklearn linear model yields a slope that is the same as that calculated by plotly express above: <strong>0.098</strong> Note that the intercept is slightly shifted- but we are using a random sample here vs the enter dataframe above. We use both the model's <em>.score()</em> method and the <em>mean_absolute_error</em> package to estimate how well the linear model fits the data. The negative <strong>R<sup>2</sup></strong> value of <strong>-2.09</strong> is telling us that our model does no better than a mean baseline. Test MAE is 42.26893511094306/</p>
<p>df_dl.workload.mean(), df_dl.workload.std()  = (207.0688439849624, 134.49243473471824)</p>
<h3 id="Will-a-tree-model-perform-better?">Will a tree model perform better?<a class="anchor-link" href="#Will-a-tree-model-perform-better?">&#182;</a></h3><p>Linear regression isnt really working well for this data. Shall we see if a tree model can better handle the non-monotonic change in workload? Using the same train / test split from above, we have to shape the arrays differently before feeding them to the tree ensemble.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">

<div class="output_wrapper">
<div class="output">

<div class="output_area">




 
 
<div id="63b7ab8a-d04d-4a1d-a43b-3bd86d1ba3ed"></div>
<div class="output_subarea output_widget_view ">
<script type="text/javascript">
var element = $('#63b7ab8a-d04d-4a1d-a43b-3bd86d1ba3ed');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"model_id": "c796c63530874c7ea17473dbdb76ba9f", "version_major": 2, "version_minor": 0}
</script>
</div>

</div>

<div class="output_area">



<div class="output_text output_subarea output_execute_result">
<pre>1200</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Got-trees?-Lets-fit-some-more-models-and-try-to-see-the-forest..">Got trees? Lets fit some more models and try to see the forest..<a class="anchor-link" href="#Got-trees?-Lets-fit-some-more-models-and-try-to-see-the-forest..">&#182;</a></h2><p>It is evident that the tree-based regressor has no problem fitting the wild variance of the daily queue length (workload) ##TODO##  change to queue length? Using the slider to redraw the plot with increasing depth and watch the MAE drop. But are we done? Can this model really predict future workload? Since we did <strong>random</strong> sampling of our data we would expect the fitted model to predict well on the test sets, if we have enough data points in each set. It</p>
<h3 id="Time-Based-Split">Time-Based Split<a class="anchor-link" href="#Time-Based-Split">&#182;</a></h3><p>To test our Random Forest model further, we can create a time based split. The datetime index is sorted, we will re-train the model using the first 80% of the observations along the time axis, with the remaining 20% for test. Use the <em>training set%</em> slider to observe the effect on the prediction. If you think <em>that's bogus, use the dropdown</em>, and watch what happens when you change the split as follows: 
   out of the original train set, create a <strong><em>random</em></strong> 80% sample and use the remainder for validation. prepare to be amazed...</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Linear-Regression-in-detail:">Linear Regression in detail:<a class="anchor-link" href="#Linear-Regression-in-detail:">&#182;</a></h2><p>lets see what happens if we just treat the workload to a linear regression, with a random split:</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">

<div class="output_wrapper">
<div class="output">

<div class="output_area">




 
 
<div id="c3f0e0b8-16c7-4a1d-a175-6094f1fa33fd"></div>
<div class="output_subarea output_widget_view ">
<script type="text/javascript">
var element = $('#c3f0e0b8-16c7-4a1d-a175-6094f1fa33fd');
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"model_id": "ea10449db35c4f91b5b741f5099c8d4b", "version_major": 2, "version_minor": 0}
</script>
</div>

</div>

<div class="output_area">



<div class="output_text output_subarea output_execute_result">
<pre>&lt;function __main__.f(md, ts, tvt_split=&#39;OFF&#39;)&gt;</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Wow-this-tree-model-is-very-smart!">Wow this tree model is very smart!<a class="anchor-link" href="#Wow-this-tree-model-is-very-smart!">&#182;</a></h2><p>This is what happens when you train your model on a time series with time based split! It appears to have determined that its not worth the effort and just guess a constant! but this</p>
<h3 id="Where-do-we-go-from-here?-Lets-do-some-window-looking.">Where do we go from here? Lets do some window-looking.<a class="anchor-link" href="#Where-do-we-go-from-here?-Lets-do-some-window-looking.">&#182;</a></h3><p>Now that we have run ican see the deficiencies of regression models, we need to introduce some means to better interpret the data. If there appears to be a periodic fluctuation, we can use a <strong>rolling average</strong>, or <strong>moving mean</strong> to smooth out short term variance and visualize larger trends. Below I introduce another calculated feature of the data, the <em>time to resolution</em> or <em>ttr</em> for a specific case. This is very simple; it is the time difference between the open and close for that case. In the data its expressed as a <em>timedelta</em> object, and cast to various datatypes, such as *</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">

<div class="output_wrapper">
<div class="output">

<div class="output_area">



<div class="output_png output_subarea ">
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Rolling-Means-for-Daily-Workload">Rolling Means for Daily Workload<a class="anchor-link" href="#Rolling-Means-for-Daily-Workload">&#182;</a></h2><p>To see through some of the daily variance, and try to look for seasonal effects we can calculate the rolling means with various window sizes. after doing so we can see that there is a slight upward linear trend in the daily workload but with the extreme smoothing using a 360 day window, the red line, there also appears to be a multi-year cycle, perhaps releated to staffing and or process changes, that reduce the workload for a time before the upward trend continues.</p>
<p>In contrast, the <em>mean time to resolution (ttr) in hours</em> shows extreme fluctuations, but perhaps without a clear upward trend. We'll look at this more later</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">

<div class="output_wrapper">
<div class="output">

<div class="output_area">



<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABJUAAAGGCAYAAADCTrBMAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAMTQAADE0B0s6tTgAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+j8jraAAAgAElEQVR4nOzdd3gU5fo38O/M7qYnBCS0gAklEAgqVULREEqkSBVD7yggAUUEjj8jxSjiQc+hiYIIKNKLWOAAUjRBEBGEV0VQOkQ6hCyBlN153j82O8m29GSTzfdzXWh2Znbm2WdnN7t37vseSQghQERERERERERElA+yswdARERERERERERlD4NKRERERERERESUbwwqERERERERERFRvjGoRERERERERERE+cagEhERERERERER5RuDSkRERERERERElG8MKhERERERERERUb4xqEREROVS+/bt0b59+3zfb9WqVZAkCRcuXFCXBQcHY8SIEUU2tpJ24cIFSJKE5cuX57hdQeessPIzv8ePH0eHDh3g5+eHgIAAdO3aFSdOnMjTfZs0aYJGjRo5XH/27FlIkoRZs2blaX85+f777yFJEr7//vtC74uKj/U5b+95c9brgoiIqDRgUImIiCgfunfvjkOHDqF69erOHgpZuXv3Lrp06YKUlBSsW7cOy5YtQ/Xq1fHHH3/k6f7Dhw/Hn3/+iaNHj9pd//nnn0OSJAwbNqzQY23WrBkOHTqEZs2aFXpf5FxLlizBkiVLnD0MIiIip9A6ewBERERlSUBAAAICApw9jCIhhEBGRoazh1FkEhIScP36dezYsUMN1vTp0yfP9x88eDCmTZuGzz//HM2bN7dYJ4TAF198gbZt26JOnToFHqPRaIQQAn5+fggPDy/wfihLWloa3N3d87St+Zx3c3MrsuPnlN1GRETk6pipRERELm/9+vUIDQ2Fu7s7wsLC8OWXX9psk5qaismTJ6Nx48bw8fFBtWrV0KNHD5w6dcpiO3vlb9kdPXoUkiThq6++slk3YsQI1KxZE0aj0e59P/jgA3h5eSE9PV1d9txzz0GSJOzZs0dd9sknn0Cr1SI5OVld9sUXX+CJJ56Ah4cHKleujKFDh+Lq1asW+w8ODsaQIUOwYsUKhIaGws3NDdu3b7c7llu3bqFVq1Zo2LAhLl26ZHcbADh9+jT69OkDf39/eHp6Ijw8HDt37rTY5syZMxg6dChq164NT09P1KlTB+PHj8fdu3dt9rdgwQIEBwfDw8MDLVq0QEJCgsNjW5Nl08eav//+O8/3ya5KlSro0qUL1q9fD4PBYLHuwIEDOHfuHIYPHw7AdE516NABAQEB8PHxQdOmTfHZZ5/Z7FOSJLzxxhuYO3cuateuDTc3N/z22292y6h2796Nbt26oXr16vDy8kLjxo3xwQcf2Jwv5udx/fr1aNiwIby9vdGiRQscOHDA5vg//PADOnfujAoVKsDb2xtPPPEEPv30U4ttli1bZnHujB49Gnfu3Ml1vjIyMhAbG4vg4GC4ubkhODgYsbGxaqAyLS0NlSpVwquvvmpz340bN0KSJPz6668WY+3YsSN8fX3h7e2NZ555Br///rvF/dq3b4927drhm2++QdOmTeHu7p5jllBO5/zOnTvRunVreHp6okKFCujduzdOnz6d6+O25qhE7uuvv0ZMTAwqV66MypUrY8iQIUhKSrK4782bNzFw4ED4+fmhYsWKGDlyJL7++utcSyO3bNkCSZJw5coVddmUKVNsyli/++47SJKU52w9IiKi/GJQiYiIXNqePXswaNAghISEYOvWrZg6dSpefvllmy+PaWlp0Ov1iI2Nxfbt2/HRRx8hNTUVrVu3xrVr1/J8vObNm6Nly5ZYunSpxfKkpCRs3LgRY8aMgUajsXvfyMhIPHz4ED/99BMAU1bF999/D09PT+zbt0/dbt++fWjevDn8/PwAmIICQ4cORcOGDbF161bMnTsXu3btQkREBO7fv29xjP379+M///kPZs6ciZ07d+Lxxx+3GceFCxfQtm1bSJKEAwcO4NFHH7U73n/++Qft2rXDiRMnsHjxYmzcuBH+/v7o3r07/ve//1lsV6tWLcyfPx+7du3CjBkzsHfvXnTr1s1if59++ileeeUVREZGYtu2bRgxYgQGDhxoN/hkT8eOHREYGIiXX34Z586dy9N9rA0fPhw3btzArl27LJavXr0anp6eiI6OBgCcO3cO/fr1w5o1a7Bt2zb06NEDY8aMwccff2yzz1WrVmH79u14//33sX37dtSoUcPusc+dO4eOHTtixYoV2L59O4YPH45Zs2bhjTfesNk2ISEBH3zwAeLi4rBhwwYYjUY8++yzFkGLr776Ch07dkR6ejqWLl2Kr776CqNGjcLFixfVbf71r39hwoQJ6NSpE77++mvMmzcPO3fuRNeuXR0GP7PP1dy5czFs2DB8++23GDFiBN577z018Obu7o7o6GisW7fOZl+rV69G48aN0bRpUwDA9u3b0bFjR/j4+OCLL77A2rVrodfr8dRTT+Hy5csW9/3rr78wadIkTJw4Ebt27ULHjh1zHKe9c37nzp3o3r07fHx8sGHDBnz00Uf4/fff0a5dOyQmJua4v7x6+eWXIUkS1q5di5kzZ2LLli14+eWXLbbp27cv/ve//+Hdd9/F+vXrodPpMHHixFz3HRERAUmSbN4X7L1XVK1aFWFhYUXymIiIiGwIIiIiF9amTRvRsGFDYTQa1WWHDh0SAERERITD+xkMBpGSkiJ8fHzEf/7zH3X5ypUrBQBx/vx5dVlQUJAYPny4xTayLIsLFy6oyxYsWCA0Go24fPmyw2MajUZRsWJFMWvWLCGEEL/++quQJElMnjxZhIeHq9tVq1ZNTJ8+XR1nlSpVRPv27S32lZCQIACIBQsWWIzT09NTXL161WLb8+fPCwDik08+EcePHxfVq1cXXbt2FSkpKRbbRUREWMzZlClThEajEX///bfFvNWvX180bdrU4ePMyMhQx3fs2DH1sdesWVM888wzFtuuX79eALCYX0e+//57UaNGDfHoo4+K2rVriytXruR6H2upqamiYsWKIjo6Wl328OFD4e/vLwYMGGD3PkajUWRkZIgxY8aIxx9/3GIdAFG9enXx4MEDi+X79+8XAMT+/fvt7lNRFJGRkSHefvtt4e/vb3H+BgUFCX9/f3Hnzh112ZEjRwQAsWbNGvX+QUFBonnz5hb3ze78+fNClmUxe/Zsi+UHDhwQAMSXX35p935CCPHbb78JAGLmzJkWy+Pi4gQAceLECYt97dy5U93mxo0bQqvVivfee09dVrduXdGhQweLfd27d0888sgj4uWXX1aXRURECEmSxK+//upwbNk5OuebN28u6tWrJzIyMtRl586dE1qtVkyePNnieNnPeXvPm6Nthg0bZnHMCRMmCHd3d6EoihBCiF27dgkAYsOGDRbb9ejRI8dzw+zxxx8XI0aMEEIIcfv2bSFJknj11VdF9erV1W1atWol+vfvn+N+iIiICoOZSkRE5LKMRiOOHDmCfv36qaVRABAeHo7g4GCb7Tdu3IhWrVrB398fWq0W3t7euH//fr5LYgYMGAB/f3988skn6rKlS5eie/fuqFmzpsP7ybKMiIgINdNg3759ePzxx/H888/jl19+gV6vx8mTJ3Ht2jVERkYCMJWf3bhxA4MHD7bYV7t27RAUFIQffvjBYnl4eDiqVatm9/jx8fGIiIhQs1a8vLxyfJzx8fEIDw9HvXr11GUajQYDBw7E8ePH1fK89PR0zJkzB6GhofD09IROp8NTTz2ljh8Arly5gitXrqiZQGbPPfcctNrcW0D+9ddfePbZZ/Huu+8iISEBiqKgc+fOuHXrlrpNSEgIYmJictyPu7s7BgwYgK+//hr37t0DAHz99ddISkpSM3AAU4ndwIEDERgYCJ1OB51Oh+XLl9s9V7p06QJPT89cH8PVq1cxduxYBAUFwc3NDTqdDrGxsUhKSsKNGzcstm3dujUqVqyo3n7ssccAQC1VPH36NC5evIgxY8ZYnPvZfffdd1AUBYMHD4bBYFD/tWrVCr6+voiPj3c4VvO6IUOGWCw33zafd23btkXdunWxevVqdZv169erxwVMc3n27FmbcXh5eaF169Y24wgODkaTJk0cjs2a9TmfkpKCY8eOoX///hbnVu3atdG2bVub10xBde/e3eL2Y489hrS0NFy/fh0A8NNPP0Gj0dj0/erXr1+e9t+hQwfs378fgKnkzt/fH5MnT8bVq1fx559/Qq/X4+jRo+p7BRERUXFgUImIiFzWrVu3kJGRgapVq9qss172zTffoH///mjYsCHWrl2Lw4cP48iRIwgICEBqamq+juvh4YGRI0dixYoVMBgMSEhIwMmTJzFu3Lhc7xsZGYmffvoJDx8+xP79+xEZGYmWLVvCw8MDCQkJ2L9/P3Q6Hdq1awcAau8be1ejq1atmk1vnJyuWrdjxw7cv38fY8eOzVMg586dOw6PK4RQy9Zef/11zJo1C0OGDMH27dvx888/Y+vWrQCgzq25/5P186LVavHII4/kOpb58+ejYsWKGDZsGB599FHs27cPer0eUVFRuHfvHv755x+cPXsWzzzzTK77Gj58OFJTU7Fp0yYApqu+Va9eHZ07dwYA3L9/H507d8aJEycwd+5cJCQk4MiRIxg1ahTS0tJs9peXKwUqioKePXvi22+/RWxsLPbt24cjR46opW/W52ClSpUsbpsbVZu3u337NgDkGMQ0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/Hj069cPd+7cwfbt29GlSxd1u0WLFqFnz55444030L9/f+j1eqxbt85mf6NHj8b8+fOxb98+9OrVCytXrsS6devslmZZq169On7++Wf06dMHCxYsQJcuXTBlyhRUqFABx48fx44dO3DkyBH07dsX6enpue6vZs2a6NChA4QQGDZsmMW6gIAAfPnllzAajejXrx9ef/11jBkzxqZpdX64ublh27ZtqFatGoYNG4YJEybg6aefxr/+9a8C77NXr1747rvvAJjmtmfPnli2bJlFk/o5c+Zg2bJliI+PR3R0NHr16oX33nsPFStWzLVsatWqVZg+fTpWrFiBbt264dNPP8X06dMtStvMzE2xr127hqFDh9qs79atG+Lj45GSkoIxY8bgmWeewbRp03Dt2jWb870odOnSBdu3b0dSUhKio6Mxbtw4NGzYEAcOHECNGjWK/HiOfPnll+jSpQumT5+O6OhopKamIi4uDgDy9N5jfq/IHvyrXLmyGtRjphIRERU3STjKiSYiIqJC+e677xAVFYU9e/agY8eOzh4OEZUBMTExWLlyJe7cuaM29yYiIiqtmKlUwo4fP+7sIZRLnPeSxzl3Ds57ybM352fPnsV3332HyZMno1mzZgwoFQOe6yWPc170Vq1ahQULFmDPnj3YsWMHJk2ahI8++ggTJ05UA0qc95LHOXcOznvJ45w7h6vNO4NKJcxcJ08li/Ne8jjnzsF5L3n25jwuLg5du3aFu7s7Pv/8cyeMyvXxXC95nPOi5+3tjZUrV6JPnz7o3bs3du3ahTlz5mDOnDnqNpz3ksc5dw7Oe8njnDuHq817mQwq5RbZc+b6wkYdC7NvZz6u0r4+N8X12J39uJ059tyU1nkrzc9JXtY789iuOq8FmfNVq1bBYDDg6NGjyMjIKJZxOXu9s8eWG76v5399aZ3z4l7vzGM///zzWLVqFfR6PdLT03H69GlMnz49z1cpzG3/5XVec1tfnOd6YY9dmte76vt6YdeX5ecsN2X1cRd2Pc/1/CmTQaW//vorx/UnT5502vrc7mt9SeWi3LczH1dpX++seXf243bm2HmuO2d9TvPOeS3Yelc91wu73tlj4/t60a8vrXNe3OtL89iAsjvvzp43Z53rhT12aV7Pc730HTu39a4658W9nud6/rBRNxERERERERER5ZvW2QMoqGvXrqEsxsN8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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">

<div class="output_wrapper">
<div class="output">

<div class="output_area">



<div class="output_png output_subarea output_execute_result">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAh8AAAG0CAYAAACSbkVhAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAMTQAADE0B0s6tTgAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+j8jraAAAgAElEQVR4nO3deXiU9b3//9dkI3sCmUCgrEdSEESiRAIKhV4GsSiiiJQGK27gr9FaLVWP54gsVnE5tlIbCrHHsgge1Bo9oqBwtIoLuGD4aRVckIKaECaQhc1sn+8fNFMmC07I3J/JDM/HdXGF3PO573nf78zM/Zp7mXEZY4wAAAAsiQh2AQAA4NRC+AAAAFYRPgAAgFWEDwAAYBXhAwAAWEX4AAAAVhE+AACAVYQPwA+PPPKInn322WbT582bJ5fLFYSKmvvb3/6mefPmqaGhIdilOMLlcmnevHltnq+4uFjz5s3T/v37A7bM9li2bJlcLpe++OILq/f7fY4cOaKUlBS5XC5t27at1XEej0d33nmnBg8erISEBMXHx2vIkCH693//d5WUlHjH9e3bVy6Xq9m/UaNG2VgddHBRwS4ACAWPPPKIRo0apcmTJ/tMv/7663XhhRcGqSpff/vb3zR//nzdddddiojgfUWj4uJizZ8/X1deeaW6dOnic9s777yjnj17BqmyjqWoqEhVVVWSpBUrVujhhx9uNuaTTz7RBRdcIGOMbr75ZmVnZ0uSPvzwQy1dulQ7duxQUVGRd/z48eObhbvk5GTnVgIhg/ABtEPPnj1PuY3Xd999p06dOgW7jIAYMWJEsEtok9raWkVFRTmyt2358uXq0qWLMjMztWrVKj3wwAOKivrXJqKurk6XX365YmNj9fbbb6tr167e284//3zdcsstWrdunc8y3W53yPUYdvD2CI5pPCSxfft2jR8/XgkJCerdu7f+8pe/SJJWrlypgQMHKjExUT/+8Y/15ZdfNltGYWGhhg4dqtjYWLndbl133XXNdp//8Y9/1MiRI9WlSxelpqZqxIgRevHFF33G7Nq1Sy6XS0uXLtXdd9+t7t27KzU1VRMnTtTXX399wvXo27ev/vGPf2jVqlXeXcdXX321zzoez+Vy6a677tLDDz+sPn36KD4+XhdddJHKyspUVlamqVOnKiUlRb169dIDDzzQ7P6++uorTZ8+Xenp6erUqZOysrJ83k221uv58+dLkqKjo711NiopKdFVV10lt9utTp066cwzz9QTTzxxwmVKx/amuFwuPfvss5o5c6bS09PVrVs37+3+/H0WLVqk008/XXFxcercubOys7N91scYo9///vcaMGCAYmJi1L17d910003ed+Gtufrqq9W3b99m08eOHauxY8dKOnaI45prrpEkZWZmevuya9cuSS0fdlm/fr1GjhypuLg4paSk6NJLL9WOHTua3ceoUaO0ceNGnX322YqPj9cZZ5zxvX+n43k8Hk2fPl3Jycnq0aOHbr75Zh09etR7e+NjdvHixbr99tvVo0cPderUSRUVFSotLdWMGTO807p3766LL75YZWVlft//8b755htt3LhR06ZN0/XXX6+9e/fq5Zdf9hlTVFSk7du36/777/cJHo2ioqI0ceLEk7p/nIIM4JC5c+caSeaMM84wixYtMq+88oq59NJLjSRz5513mpEjR5qioiLz1FNPme7du5vhw4f7zH/HHXeYqKgo8+tf/9q8/PLL5vHHHzc9evQww4cPN3V1dd5xs2fPNn/+85/Nxo0bzfr1682NN95oJJl169Z5x3z11VdGkunTp4/52c9+Zl566SWzbNkyk5aWZsaMGXPC9di6davJyMgw48ePN++884555513zBdffOGzjseTZHr37m0mTJhg1q5da/77v//bJCUlmfHjx5tzzz3X3HPPPWbDhg1m1qxZRpJ58cUXvfPu3r3bpKenm8GDB5uVK1ea9evXm2uuuca4XC7z/PPPt1rjnj17zHXXXWckmTfffNNbpzHGHDx40GRmZhq3222WLl1qXnrpJZOXl2ckmaVLl55w3V977TUjyfTo0cNcd911Zt26daaoqMjvv88TTzxhIiMjzfz5882rr75qXnzxRbNw4ULz5z//2Xsfd955p5FkbrzxRrN+/Xrzu9/9ziQkJJhRo0aZ+vp6n77OnTvX+/uMGTNMnz59mtU8ZswY79+0rKzM3HXXXUaSefrpp719OXr0aIvLXLdunYmIiDC5ubnm+eefN6tWrTKnnXaacbvd5uuvv/a5j4yMDDNo0CCzcuVKs27dOpObm2siIyPN559/fsKe/uUvfzGSTP/+/c2cOXPMhg0bzIIFC0xERIS5++67veMaH7M9evQwkyZNMi+88IJ57rnnzOHDh01ubq7JzMw0TzzxhHn99dfNU089ZW644Qbz1VdfnfC+W3P//fcbSWbz5s2moqLCxMbGmqlTp/qMmTlzpomMjDSHDh3ya5l9+vQxeXl5pra21udfQ0PDSdWI8EL4gGMaN8zLly/3Ttu/f7+JjIw0Xbp0MZWVld7pixYtMpLMrl27jDHHXngjIiLM/PnzfZb55ptvGkneDWBT9fX1pra21owbN85ccskl3umNL+RNg8ZDDz1kJJlvvvnmhOvSp08fM3369FbX8XiSTGZmpqmtrfVOu/XWW40kc88993in1dbWmvT0dHP11Vd7p1177bXG7XYbj8fjs8zc3FwzdOjQE9bYWMvx92uMMY8++qiRZF577TWf6eeff75JT0/3CXJNNYaPSy+91Ge6v3+fG2+80Zx11lmtLr+8vNzExMSYGTNm+ExfuXKlkeQTuE4mfBjzr419S6Gg6TKHDRtm+vfv79PDnTt3mqioKHPrrbf63EdUVJT57LPPvNP27t1rIiIizL333tvq+h5fz/FBwxhjLrroIpOZmen9vfExe9ZZZzXbYCckJJhFixad8H7a4vTTTzcDBgzw/j5t2jQTGxtrDhw44J124YUXmoyMDL+X2adPHyOp2b8NGzYErG6ELg67wHE/+clPvP/v3LmzunbtqhEjRviceDZw4EBJ0p49eyRJGzZsUENDg6ZPn666ujrvv5ycHCUlJemNN97wzvvBBx/o4osvVrdu3RQVFaXo6Ght2LCh2a5ySZowYYLP70OGDJEk7d69O3ArLGncuHE+x8sb12/8+PHeaVFRUerfv793naVju/wnTJiglJQUn/UeP368tm3b9r2HIlryxhtv6Ac/+IH3UESjK6+8Uvv27dMnn3zyvcu47LLLfH739+9zzjnnqLi4WL/85S+1ceNGHT582Gc5mzdvVk1Nja688kqf6dOmTVNUVJRef/31Nq/vyTp06JC2bt2qn/70pz5/u379+um8885rVktmZqYyMzO9v3ft2lVdu3b1+7F00UUX+fw+ZMiQFue99NJLmx3aO+ecc/TQQw9p0aJF+uijj2Ta8eXk7733nj799FP9/Oc/906bMWOGjh49qjVr1pz0cqVjz/333nvP519OTk67lonwQPiA4zp37uzze0xMTIvTJHmPeTceu+7fv7+io6N9/lVXV6u8vFzSsbBy/vnna//+/Xr00Uf19ttv67333tOFF17oc/y8UdOrHRpPnGxpbHu0tn4tTT/+vsvKyrRixYpm63zbbbdJkne922L//v3q3r17s+kZGRne279P0/n9/ftcddVV+tOf/qQtW7Zo/Pjx6tKliyZPnuw956LxvpsuPyoqSmlpaX7VFigHDhyQMabVXjWtpeljSTr2ePL3sdTSY/G7775rNq6letasWaNLLrlEDz74oM4880z94Ac/0IIFC07qMuvly5dLkiZOnKiKigpVVFTonHPOUXp6ulasWOEd16tXL+3bt69ZgDyRLl26KDs72+dfUlJSm2tE+OFqF3RIaWlpkqRXXnml2Qb7+NvXr1+vyspKPfXUUz5XnbTlBbIjSUtL0+jRo3XHHXe0eHuPHj3avMwuXbq0uBeotLTUe/v3afrO29+/j8vl0g033KAbbrhBBw4c0CuvvKLZs2frpz/9qbZs2eK979LSUg0ePNg7f11dncrLy09YW2xsrGpqappNLy8v995/W3Tu3Fkul8vbl+OVlpb61ScntHRlS9euXVVQUKCCggLt2LFDy5cv19y5c5Wenq5f/OIXfi+7pqZGTz75pCRp6NChzW7ft2+fPv/8c2VmZio3N1ePPfaY1q1bp8svv/zkVwgQ4QMd1Lhx4xQREaHdu3dr3LhxrY5rDBnR0dHeaZ999pneeuutgF4C26lTJx05ciRgy2vNhRdeqHfeeUeDBw9WXFxcm+Zt3Itz5MgRn3eXY8aM0dNPP6233npL5513nnf66tWr1bVrVw0aNKjNdfr79zle586dvaFj6dKlko5d6hoTE6P/+Z//0fnnn+8du2bNGtXV1TU7VHS8Pn36aO/evdq3b5/S09MlSV9++aV27Nihc8891zvu+L6cSEJCgoYNG6ann35a8+bNU2RkpCTpH//4h95++2398pe/9Gs9bRswYIDuu+8+LVmyRB9//HGb5l27dq3279+vuXPnNuv13r17NW3aNK1YsUL33HOPJk+erAEDBuiOO+7Qj370I2/PG9XV1enll19udjgJaAnhAx3SaaedpjvuuEM33XSTduzYoTFjxig2NlZ79uzRhg0bdP311+vHP/6xcnNzFRUVpauuukqzZ89WSUmJ5s6dq969ewf0kz4HDRqkTZs2ae3atcrIyJDb7W7xMs/2WrBggYYPH64f/ehHuummm9S3b18dOHBA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
"
>
</div>

</div>

<div class="output_area">



<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAh8AAAG0CAYAAACSbkVhAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAMTQAADE0B0s6tTgAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+j8jraAAAgAElEQVR4nO3deXiU9b3//9dkI3sCmUCgrEdSEESiRAIKhV4GsSiiiJQGK27gr9FaLVWP54gsVnE5tlIbCrHHsgge1Bo9oqBwtIoLuGD4aRVckIKaECaQhc1sn+8fNFMmC07I3J/JDM/HdXGF3PO573nf78zM/Zp7mXEZY4wAAAAsiQh2AQAA4NRC+AAAAFYRPgAAgFWEDwAAYBXhAwAAWEX4AAAAVhE+AACAVYQPwA+PPPKInn322WbT582bJ5fLFYSKmvvb3/6mefPmqaGhIdilOMLlcmnevHltnq+4uFjz5s3T/v37A7bM9li2bJlcLpe++OILq/f7fY4cOaKUlBS5XC5t27at1XEej0d33nmnBg8erISEBMXHx2vIkCH693//d5WUlHjH9e3bVy6Xq9m/UaNG2VgddHBRwS4ACAWPPPKIRo0apcmTJ/tMv/7663XhhRcGqSpff/vb3zR//nzdddddiojgfUWj4uJizZ8/X1deeaW6dOnic9s777yjnj17BqmyjqWoqEhVVVWSpBUrVujhhx9uNuaTTz7RBRdcIGOMbr75ZmVnZ0uSPvzwQy1dulQ7duxQUVGRd/z48eObhbvk5GTnVgIhg/ABtEPPnj1PuY3Xd999p06dOgW7jIAYMWJEsEtok9raWkVFRTmyt2358uXq0qWLMjMztWrVKj3wwAOKivrXJqKurk6XX365YmNj9fbbb6tr167e284//3zdcsstWrdunc8y3W53yPUYdvD2CI5pPCSxfft2jR8/XgkJCerdu7f+8pe/SJJWrlypgQMHKjExUT/+8Y/15ZdfNltGYWGhhg4dqtjYWLndbl133XXNdp//8Y9/1MiRI9WlSxelpqZqxIgRevHFF33G7Nq1Sy6XS0uXLtXdd9+t7t27KzU1VRMnTtTXX399wvXo27ev/vGPf2jVqlXeXcdXX321zzoez+Vy6a677tLDDz+sPn36KD4+XhdddJHKyspUVlamqVOnKiUlRb169dIDDzzQ7P6++uorTZ8+Xenp6erUqZOysrJ83k221uv58+dLkqKjo711NiopKdFVV10lt9utTp066cwzz9QTTzxxwmVKx/amuFwuPfvss5o5c6bS09PVrVs37+3+/H0WLVqk008/XXFxcercubOys7N91scYo9///vcaMGCAYmJi1L17d910003ed+Gtufrqq9W3b99m08eOHauxY8dKOnaI45prrpEkZWZmevuya9cuSS0fdlm/fr1GjhypuLg4paSk6NJLL9WOHTua3ceoUaO0ceNGnX322YqPj9cZZ5zxvX+n43k8Hk2fPl3Jycnq0aOHbr75Zh09etR7e+NjdvHixbr99tvVo0cPderUSRUVFSotLdWMGTO807p3766LL75YZWVlft//8b755htt3LhR06ZN0/XXX6+9e/fq5Zdf9hlTVFSk7du36/777/cJHo2ioqI0ceLEk7p/nIIM4JC5c+caSeaMM84wixYtMq+88oq59NJLjSRz5513mpEjR5qioiLz1FNPme7du5vhw4f7zH/HHXeYqKgo8+tf/9q8/PLL5vHHHzc9evQww4cPN3V1dd5xs2fPNn/+85/Nxo0bzfr1682NN95oJJl169Z5x3z11VdGkunTp4/52c9+Zl566SWzbNkyk5aWZsaMGXPC9di6davJyMgw48ePN++884555513zBdffOGzjseTZHr37m0mTJhg1q5da/77v//bJCUlmfHjx5tzzz3X3HPPPWbDhg1m1qxZRpJ58cUXvfPu3r3bpKenm8GDB5uVK1ea9evXm2uuuca4XC7z/PPPt1rjnj17zHXXXWckmTfffNNbpzHGHDx40GRmZhq3222WLl1qXnrpJZOXl2ckmaVLl55w3V977TUjyfTo0cNcd911Zt26daaoqMjvv88TTzxhIiMjzfz5882rr75qXnzxRbNw4ULz5z//2Xsfd955p5FkbrzxRrN+/Xrzu9/9ziQkJJhRo0aZ+vp6n77OnTvX+/uMGTNMnz59mtU8ZswY79+0rKzM3HXXXUaSefrpp719OXr0aIvLXLdunYmIiDC5ubnm+eefN6tWrTKnnXaacbvd5uuvv/a5j4yMDDNo0CCzcuVKs27dOpObm2siIyPN559/fsKe/uUvfzGSTP/+/c2cOXPMhg0bzIIFC0xERIS5++67veMaH7M9evQwkyZNMi+88IJ57rnnzOHDh01ubq7JzMw0TzzxhHn99dfNU089ZW644Qbz1VdfnfC+W3P//fcbSWbz5s2moqLCxMbGmqlTp/qMmTlzpomMjDSHDh3ya5l9+vQxeXl5pra21udfQ0PDSdWI8EL4gGMaN8zLly/3Ttu/f7+JjIw0Xbp0MZWVld7pixYtMpLMrl27jDHHXngjIiLM/PnzfZb55ptvGkneDWBT9fX1pra21owbN85ccskl3umNL+RNg8ZDDz1kJJlvvvnmhOvSp08fM3369FbX8XiSTGZmpqmtrfVOu/XWW40kc88993in1dbWmvT0dHP11Vd7p1177bXG7XYbj8fjs8zc3FwzdOjQE9bYWMvx92uMMY8++qiRZF577TWf6eeff75JT0/3CXJNNYaPSy+91Ge6v3+fG2+80Zx11lmtLr+8vNzExMSYGTNm+ExfuXKlkeQTuE4mfBjzr419S6Gg6TKHDRtm+vfv79PDnTt3mqioKHPrrbf63EdUVJT57LPPvNP27t1rIiIizL333tvq+h5fz/FBwxhjLrroIpOZmen9vfExe9ZZZzXbYCckJJhFixad8H7a4vTTTzcDBgzw/j5t2jQTGxtrDhw44J124YUXmoyMDL+X2adPHyOp2b8NGzYErG6ELg67wHE/+clPvP/v3LmzunbtqhEjRviceDZw4EBJ0p49eyRJGzZsUENDg6ZPn666ujrvv5ycHCUlJemNN97wzvvBBx/o4osvVrdu3RQVFaXo6Ght2LCh2a5ySZowYYLP70OGDJEk7d69O3ArLGncuHE+x8sb12/8+PHeaVFRUerfv793naVju/wnTJiglJQUn/UeP368tm3b9r2HIlryxhtv6Ac/+IH3UESjK6+8Uvv27dMnn3zyvcu47LLLfH739+9zzjnnqLi4WL/85S+1ceNGHT582Gc5mzdvVk1Nja688kqf6dOmTVNUVJRef/31Nq/vyTp06JC2bt2qn/70pz5/u379+um8885rVktmZqYyMzO9v3ft2lVdu3b1+7F00UUX+fw+ZMiQFue99NJLmx3aO+ecc/TQQw9p0aJF+uijj2Ta8eXk7733nj799FP9/Oc/906bMWOGjh49qjVr1pz0cqVjz/333nvP519OTk67lonwQPiA4zp37uzze0xMTIvTJHmPeTceu+7fv7+io6N9/lVXV6u8vFzSsbBy/vnna//+/Xr00Uf19ttv67333tOFF17oc/y8UdOrHRpPnGxpbHu0tn4tTT/+vsvKyrRixYpm63zbbbdJkne922L//v3q3r17s+kZGRne279P0/n9/ftcddVV+tOf/qQtW7Zo/Pjx6tKliyZPnuw956LxvpsuPyoqSmlpaX7VFigHDhyQMabVXjWtpeljSTr2ePL3sdTSY/G7775rNq6letasWaNLLrlEDz74oM4880z94Ac/0IIFC07qMuvly5dLkiZOnKiKigpVVFTonHPOUXp6ulasWOEd16tXL+3bt69ZgDyRLl26KDs72+dfUlJSm2tE+OFqF3RIaWlpkqRXXnml2Qb7+NvXr1+vyspKPfXUUz5XnbTlBbIjSUtL0+jRo3XHHXe0eHuPHj3avMwuXbq0uBeotLTUe/v3afrO29+/j8vl0g033KAbbrhBBw4c0CuvvKLZs2frpz/9qbZs2eK979LSUg0ePNg7f11dncrLy09YW2xsrGpqappNLy8v995/W3Tu3Fkul8vbl+OVlpb61ScntHRlS9euXVVQUKCCggLt2LFDy5cv19y5c5Wenq5f/OIXfi+7pqZGTz75pCRp6NChzW7ft2+fPv/8c2VmZio3N1ePPfaY1q1bp8svv/zkVwgQ4QMd1Lhx4xQREaHdu3dr3LhxrY5rDBnR0dHeaZ999pneeuutgF4C26lTJx05ciRgy2vNhRdeqHfeeUeDBw9WXFxcm+Zt3Itz5MgRn3eXY8aM0dNPP6233npL5513nnf66tWr1bVrVw0aNKjNdfr79zle586dvaFj6dKlko5d6hoTE6P/+Z//0fnnn+8du2bNGtXV1TU7VHS8Pn36aO/evdq3b5/S09MlSV9++aV27Nihc8891zvu+L6cSEJCgoYNG6ann35a8+bNU2RkpCTpH//4h95++2398pe/9Gs9bRswYIDuu+8+LVmyRB9//HGb5l27dq3279+vuXPnNuv13r17NW3aNK1YsUL33HOPJk+erAEDBuiOO+7Qj370I2/PG9XV1enll19udjgJaAnhAx3SaaedpjvuuEM33XSTduzYoTFjxig2NlZ79uzRhg0bdP311+vHP/6xcnNzFRUVpauuukqzZ89WSUmJ5s6dq969ewf0kz4HDRqkTZs2ae3atcrIyJDb7W7xMs/2WrBggYYPH64f/ehHuummm9S3b18dOHBA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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">

<div class="output_wrapper">
<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">
<pre>p value is 0.41090577512001386
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Since the p value islarger than the significance level of 0.05, we can reject the null hypothesis that the time series data is non-stationary. Thus, the time series data is stationary.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">

</div>
<div class="cell border-box-sizing code_cell rendered">

<div class="output_wrapper">
<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">
<pre>p value is 21.144166666666667
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">

<div class="output_wrapper">
<div class="output">

<div class="output_area">



<div class="output_text output_subarea output_execute_result">
<pre>Category                    Request Type                                      
311 External Request        Damaged_Property                                      22 days 05:49:23.230769
                            Garbage                                              152 days 17:17:50.750000
                            Graffiti                                              59 days 17:47:51.275590
                            Human/Animal Waste                                            0 days 09:56:30
                            Illegal_Posting                                        6 days 16:38:17.500000
                            Other                                                  3 days 03:11:18.111111
                            Sewer_Water_Storm_Conditions                                  0 days 01:07:34
                            Sidewalk_or_Curb_Issues                               82 days 23:08:25.888888
                            Utility Lines/Wires                                  176 days 15:22:11.750000
                            Value5                                                        0 days 00:36:39
Abandoned Vehicle           Abandoned Vehicle -                                    5 days 06:25:14.260869
                            Abandoned Vehicle - Car2door                           5 days 01:01:20.125000
                            Abandoned Vehicle - Car4door                           5 days 01:44:22.452380
                            Abandoned Vehicle - DeliveryTruck                             4 days 22:35:36
                            Abandoned Vehicle - Motorcycle                         4 days 23:16:52.512820
                            Abandoned Vehicle - Other                              5 days 00:27:46.231707
                            Abandoned Vehicle - PickupTruck                        5 days 15:03:24.440217
                            Abandoned Vehicle - SUV                                4 days 23:49:26.029411
                            Abandoned Vehicle - Trailer                            8 days 11:14:50.588235
                            Abandoned Vehicles                                     4 days 10:35:59.627692
                            SSP Abandoned Vehicles                                 3 days 19:19:46.500000
Blocked Street or SideWalk  Blocked_Parking_Space_or_Strip                        41 days 06:06:05.009615
                            Blocked_Sidewalk                                      63 days 17:31:38.783919
Damaged Property            Damaged Benches_on_Sidewalk                          210 days 00:42:46.352941
                            Damaged Bike_Rack                                     14 days 16:20:10.142857
                            Damaged Fire_Police_Callbox                           47 days 13:44:53.545454
                            Damaged Kiosk_Public_Toilet                          210 days 10:47:42.500000
                            Damaged News_Rack                                     63 days 19:27:11.296296
                            Damaged Parking_Meter                                 29 days 19:02:16.024793
                            Damaged Traffic_Signal                               179 days 11:13:51.232142
                            Damaged Transit_Shelter_Ad_Kiosk                       0 days 21:10:05.416666
                            Damaged Transit_Shelter_Platform                      26 days 09:16:23.854166
                            Damaged Transit_Shelter_Platform_Hazardous            17 days 14:17:47.700000
                            Damaged other                                         64 days 04:30:36.750000
Encampments                 Encampment Reports                                     9 days 09:34:36.290220
                            Encampment items                                       4 days 01:57:16.049645
Graffiti                    Graffiti on ATT_Property                              17 days 05:22:59.238805
                            Graffiti on Bike_rack                                        82 days 05:59:04
                            Graffiti on Bridge                                   103 days 01:44:55.250000
                            Graffiti on Building_commercial                       38 days 01:46:18.186602
                            Graffiti on Building_other                           233 days 16:38:11.049904
                            Graffiti on Building_residential                      32 days 01:46:40.841317
                            Graffiti on City_receptacle                           14 days 07:21:46.980000
                            Graffiti on Fire_Police_Callbox                       69 days 22:51:28.395061
                            Graffiti on Fire_call_box                             23 days 02:15:08.125000
                            Graffiti on Fire_hydrant                              41 days 11:13:08.983333
                            Graffiti on Fire_hydrant_puc                                 30 days 03:11:00
                            Graffiti on Mail_box                                 178 days 07:11:02.402985
                            Graffiti on News_rack                                 52 days 22:34:16.893442
                            Graffiti on Other                                      6 days 04:14:41.105263
                            Graffiti on Other_enter_additional_details_below      39 days 03:42:13.963320
                            Graffiti on Other_for_Parks_ONLY                       5 days 05:39:50.333333
                            Graffiti on Parking_meter                             34 days 23:13:01.350000
                            Graffiti on Pay_phone                                315 days 18:22:16.800000
                            Graffiti on Pole                                      22 days 00:22:35.724543
                            Graffiti on Private Property                          44 days 02:36:16.181818
                            Graffiti on Public Property                                   0 days 00:33:23
                            Graffiti on Sidewalk_in_front_of_property             35 days 06:58:37.156342
                            Graffiti on Sidewalk_structure                        26 days 05:41:34.362694
                            Graffiti on Sign                                     115 days 21:22:17.610169
                            Graffiti on Signal_box                                15 days 03:38:18.579487
                            Graffiti on Street                                    39 days 09:00:18.452631
                            Graffiti on Transit_Shelter_Platform                  34 days 23:05:53.637168
Homeless Concerns           Individual Concerns                                    5 days 13:17:24.202749
Illegal Postings            Illegal Posting - Affixed_Improperly                  75 days 22:11:29.625000
                            Illegal Posting - Multiple_Postings                          16 days 19:52:09
                            Illegal Posting - No_Posting_Date                      5 days 06:49:28.981481
                            Illegal Posting - Posted_Over_70_Days                  9 days 02:07:40.333333
                            Illegal Posting - Posted_on_Directional_Sign                  0 days 16:51:51
                            Illegal Posting - Posted_on_Traffic_Light              6 days 05:12:05.387096
                            Illegal Posting - Posting_Too_High_on_Pole                    1 days 01:03:45
                            Illegal Postings - Affixed_Improperly                 33 days 07:10:58.814814
                            Illegal Postings - Multiple_Postings                  27 days 13:24:58.057142
                            Illegal Postings - No_Posting_Date                     4 days 11:07:49.375000
                            Illegal Postings - Posted_Over_70_Days                11 days 14:26:31.033333
                            Illegal Postings - Posted_on_Directional_Sign          1 days 18:05:32.800000
                            Illegal Postings - Posted_on_Historic_Street_Light            2 days 03:12:32
                            Illegal Postings - Posted_on_Traffic_Light            55 days 19:28:55.790697
                            Illegal Postings - Posting_Too_High_on_Pole                  28 days 02:48:01
                            Illegal Postings - Posting_Too_Large_in_Size           2 days 09:30:54.250000
Litter Receptacles          Add_remove_garbage_can                                77 days 01:59:57.339622
                            Cans_Left_Out_24x7                                    49 days 19:52:47.262500
                            City_Can_Other                                        13 days 06:55:17.062500
                            City_Can_Removed                                              8 days 07:27:49
                            Damaged_City_Can                                      16 days 04:57:57.614285
                            Debris_Box                                                  597 days 08:14:22
                            Debris_box_maintenance_overflowing                     3 days 19:52:44.500000
                            Door_Lock_issues                                       1 days 18:43:43.750000
                            Door_lock_issue                                       16 days 01:29:28.437500
                            Liner_issue_damaged_missing                                   2 days 20:34:45
Name: ttr, dtype: timedelta64[ns]</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">

</div>
<div class="cell border-box-sizing code_cell rendered">

<div class="output_wrapper">
<div class="output">

<div class="output_area">


<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>CaseID</th>
      <th>ttr</th>
    </tr>
    <tr>
      <th>Neighborhood</th>
      <th>Request Details</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Alamo Square</th>
      <th>Human or Animal Waste</th>
      <td>7</td>
      <td>0 days 10:39:00.142857</td>
    </tr>
    <tr>
      <th>Apparel City</th>
      <th>Human or Animal Waste</th>
      <td>4</td>
      <td>0 days 03:51:19.250000</td>
    </tr>
    <tr>
      <th>Ashbury Heights</th>
      <th>Human or Animal Waste</th>
      <td>1</td>
      <td>2 days 21:04:00</td>
    </tr>
    <tr>
      <th>Bayview</th>
      <th>Human or Animal Waste</th>
      <td>15</td>
      <td>2 days 07:26:19.200000</td>
    </tr>
    <tr>
      <th>Bernal Heights</th>
      <th>Human or Animal Waste</th>
      <td>8</td>
      <td>0 days 21:51:12.375000</td>
    </tr>
    <tr>
      <th>Bret Harte</th>
      <th>Human or Animal Waste</th>
      <td>7</td>
      <td>2 days 15:28:54.428571</td>
    </tr>
    <tr>
      <th>Buena Vista</th>
      <th>Human or Animal Waste</th>
      <td>4</td>
      <td>1 days 00:59:01.500000</td>
    </tr>
    <tr>
      <th>Castro</th>
      <th>Human or Animal Waste</th>
      <td>15</td>
      <td>1 days 06:59:20.800000</td>
    </tr>
    <tr>
      <th>Cathedral Hill</th>
      <th>Human or Animal Waste</th>
      <td>26</td>
      <td>1 days 12:31:10.538461</td>
    </tr>
    <tr>
      <th>Cayuga</th>
      <th>Human or Animal Waste</th>
      <td>4</td>
      <td>3 days 05:36:54.750000</td>
    </tr>
    <tr>
      <th>Central Waterfront</th>
      <th>Human or Animal Waste</th>
      <td>5</td>
      <td>5 days 11:02:18.200000</td>
    </tr>
    <tr>
      <th>Chinatown</th>
      <th>Human or Animal Waste</th>
      <td>26</td>
      <td>105 days 09:08:40.576923</td>
    </tr>
    <tr>
      <th>Civic Center</th>
      <th>Human or Animal Waste</th>
      <td>58</td>
      <td>3 days 12:56:02.086206</td>
    </tr>
    <tr>
      <th>Cole Valley</th>
      <th>Human or Animal Waste</th>
      <td>4</td>
      <td>1 days 01:10:10.500000</td>
    </tr>
    <tr>
      <th>Corona Heights</th>
      <th>Human or Animal Waste</th>
      <td>2</td>
      <td>0 days 22:01:11</td>
    </tr>
    <tr>
      <th>Crocker Amazon</th>
      <th>Human or Animal Waste</th>
      <td>1</td>
      <td>1 days 16:55:00</td>
    </tr>
    <tr>
      <th>Dogpatch</th>
      <th>Human or Animal Waste</th>
      <td>2</td>
      <td>5 days 22:59:01</td>
    </tr>
    <tr>
      <th>Dolores Heights</th>
      <th>Human or Animal Waste</th>
      <td>14</td>
      <td>1 days 07:56:28.714285</td>
    </tr>
    <tr>
      <th>Downtown / Union Square</th>
      <th>Human or Animal Waste</th>
      <td>12</td>
      <td>2 days 05:32:20.416666</td>
    </tr>
    <tr>
      <th>Duboce Triangle</th>
      <th>Human or Animal Waste</th>
      <td>10</td>
      <td>35 days 02:07:01.100000</td>
    </tr>
    <tr>
      <th>Eureka Valley</th>
      <th>Human or Animal Waste</th>
      <td>7</td>
      <td>1 days 18:41:07.714285</td>
    </tr>
    <tr>
      <th>Excelsior</th>
      <th>Human or Animal Waste</th>
      <td>3</td>
      <td>0 days 14:28:31.666666</td>
    </tr>
    <tr>
      <th>Fairmount</th>
      <th>Human or Animal Waste</th>
      <td>1</td>
      <td>0 days 01:05:00</td>
    </tr>
    <tr>
      <th>Financial District</th>
      <th>Human or Animal Waste</th>
      <td>16</td>
      <td>3 days 12:36:55.375000</td>
    </tr>
    <tr>
      <th>Fisherman's Wharf</th>
      <th>Human or Animal Waste</th>
      <td>2</td>
      <td>34 days 06:31:17</td>
    </tr>
    <tr>
      <th>Glen Park</th>
      <th>Human or Animal Waste</th>
      <td>1</td>
      <td>1061 days 08:50:09</td>
    </tr>
    <tr>
      <th>Golden Gate Heights</th>
      <th>Human or Animal Waste</th>
      <td>1</td>
      <td>1 days 01:34:00</td>
    </tr>
    <tr>
      <th>Golden Gate Park</th>
      <th>Human or Animal Waste</th>
      <td>2</td>
      <td>3 days 07:41:01.500000</td>
    </tr>
    <tr>
      <th>Haight Ashbury</th>
      <th>Human or Animal Waste</th>
      <td>16</td>
      <td>44 days 04:29:10</td>
    </tr>
    <tr>
      <th>Hayes Valley</th>
      <th>Human or Animal Waste</th>
      <td>16</td>
      <td>5 days 03:58:29.562500</td>
    </tr>
    <tr>
      <th>Holly Park</th>
      <th>Human or Animal Waste</th>
      <td>3</td>
      <td>0 days 22:37:00</td>
    </tr>
    <tr>
      <th>Hunters Point</th>
      <th>Human or Animal Waste</th>
      <td>4</td>
      <td>0 days 10:57:15</td>
    </tr>
    <tr>
      <th>India Basin</th>
      <th>Human or Animal Waste</th>
      <td>2</td>
      <td>3 days 18:39:39</td>
    </tr>
    <tr>
      <th>Ingleside</th>
      <th>Human or Animal Waste</th>
      <td>6</td>
      <td>1 days 19:05:51.666666</td>
    </tr>
    <tr>
      <th>Inner Richmond</th>
      <th>Human or Animal Waste</th>
      <td>18</td>
      <td>2 days 23:13:12.277777</td>
    </tr>
    <tr>
      <th>Inner Sunset</th>
      <th>Human or Animal Waste</th>
      <td>3</td>
      <td>0 days 12:16:19</td>
    </tr>
    <tr>
      <th>Japantown</th>
      <th>Human or Animal Waste</th>
      <td>5</td>
      <td>1 days 00:46:09.800000</td>
    </tr>
    <tr>
      <th>Lakeshore</th>
      <th>Human or Animal Waste</th>
      <td>1</td>
      <td>0 days 17:48:00</td>
    </tr>
    <tr>
      <th>Laurel Heights / Jordan Park</th>
      <th>Human or Animal Waste</th>
      <td>4</td>
      <td>1 days 18:38:52.250000</td>
    </tr>
    <tr>
      <th>Lone Mountain</th>
      <th>Human or Animal Waste</th>
      <td>3</td>
      <td>5 days 20:57:00</td>
    </tr>
    <tr>
      <th>Lower Haight</th>
      <th>Human or Animal Waste</th>
      <td>22</td>
      <td>3 days 03:12:14.681818</td>
    </tr>
    <tr>
      <th>Lower Nob Hill</th>
      <th>Human or Animal Waste</th>
      <td>69</td>
      <td>6 days 11:48:26.956521</td>
    </tr>
    <tr>
      <th>Lower Pacific Heights</th>
      <th>Human or Animal Waste</th>
      <td>5</td>
      <td>2 days 00:53:36.400000</td>
    </tr>
    <tr>
      <th>Marina</th>
      <th>Human or Animal Waste</th>
      <td>9</td>
      <td>1 days 15:02:09.444444</td>
    </tr>
    <tr>
      <th>Merced Heights</th>
      <th>Human or Animal Waste</th>
      <td>1</td>
      <td>0 days 17:03:20</td>
    </tr>
    <tr>
      <th>Mint Hill</th>
      <th>Human or Animal Waste</th>
      <td>7</td>
      <td>2 days 14:54:22.428571</td>
    </tr>
    <tr>
      <th>Mission</th>
      <th>Human or Animal Waste</th>
      <td>207</td>
      <td>3 days 04:59:57.543689</td>
    </tr>
    <tr>
      <th>Mission Bay</th>
      <th>Human or Animal Waste</th>
      <td>8</td>
      <td>1 days 19:50:40.375000</td>
    </tr>
    <tr>
      <th>Mission Dolores</th>
      <th>Human or Animal Waste</th>
      <td>57</td>
      <td>4 days 14:04:03.736842</td>
    </tr>
    <tr>
      <th>Mission Terrace</th>
      <th>Human or Animal Waste</th>
      <td>5</td>
      <td>1 days 16:00:51.800000</td>
    </tr>
    <tr>
      <th>Mt. Davidson Manor</th>
      <th>Human or Animal Waste</th>
      <td>1</td>
      <td>1 days 02:22:43</td>
    </tr>
    <tr>
      <th>Nob Hill</th>
      <th>Human or Animal Waste</th>
      <td>18</td>
      <td>0 days 23:38:52.277777</td>
    </tr>
    <tr>
      <th>Noe Valley</th>
      <th>Human or Animal Waste</th>
      <td>3</td>
      <td>1 days 08:28:00</td>
    </tr>
    <tr>
      <th>North Beach</th>
      <th>Human or Animal Waste</th>
      <td>29</td>
      <td>20 days 04:42:02.931034</td>
    </tr>
    <tr>
      <th>Northern Waterfront</th>
      <th>Human or Animal Waste</th>
      <td>2</td>
      <td>0 days 01:21:02.500000</td>
    </tr>
    <tr>
      <th>Oceanview</th>
      <th>Human or Animal Waste</th>
      <td>2</td>
      <td>1 days 23:49:53</td>
    </tr>
    <tr>
      <th>Outer Mission</th>
      <th>Human or Animal Waste</th>
      <td>1</td>
      <td>0 days 02:09:00</td>
    </tr>
    <tr>
      <th>Outer Richmond</th>
      <th>Human or Animal Waste</th>
      <td>13</td>
      <td>1 days 20:47:46.307692</td>
    </tr>
    <tr>
      <th>Outer Sunset</th>
      <th>Human or Animal Waste</th>
      <td>14</td>
      <td>3 days 05:56:35.857142</td>
    </tr>
    <tr>
      <th>Pacific Heights</th>
      <th>Human or Animal Waste</th>
      <td>15</td>
      <td>1 days 11:47:34.800000</td>
    </tr>
    <tr>
      <th>Panhandle</th>
      <th>Human or Animal Waste</th>
      <td>6</td>
      <td>3 days 09:55:39</td>
    </tr>
    <tr>
      <th>Parkside</th>
      <th>Human or Animal Waste</th>
      <td>1</td>
      <td>0 days 00:33:00</td>
    </tr>
    <tr>
      <th>Peralta Heights</th>
      <th>Human or Animal Waste</th>
      <td>1</td>
      <td>1 days 02:41:00</td>
    </tr>
    <tr>
      <th>Polk Gulch</th>
      <th>Human or Animal Waste</th>
      <td>19</td>
      <td>3 days 04:10:56.578947</td>
    </tr>
    <tr>
      <th>Portola</th>
      <th>Human or Animal Waste</th>
      <td>4</td>
      <td>4 days 01:28:24</td>
    </tr>
    <tr>
      <th>Potrero Hill</th>
      <th>Human or Animal Waste</th>
      <td>29</td>
      <td>1 days 18:29:52.965517</td>
    </tr>
    <tr>
      <th>Presidio Terrace</th>
      <th>Human or Animal Waste</th>
      <td>2</td>
      <td>4 days 21:30:12</td>
    </tr>
    <tr>
      <th>Produce Market</th>
      <th>Human or Animal Waste</th>
      <td>4</td>
      <td>4 days 10:11:05.750000</td>
    </tr>
    <tr>
      <th>Rincon Hill</th>
      <th>Human or Animal Waste</th>
      <td>6</td>
      <td>2 days 03:13:58.833333</td>
    </tr>
    <tr>
      <th>Russian Hill</th>
      <th>Human or Animal Waste</th>
      <td>8</td>
      <td>8 days 21:38:34.125000</td>
    </tr>
    <tr>
      <th>Showplace Square</th>
      <th>Human or Animal Waste</th>
      <td>25</td>
      <td>5 days 01:06:10.040000</td>
    </tr>
    <tr>
      <th>Silver Terrace</th>
      <th>Human or Animal Waste</th>
      <td>1</td>
      <td>883 days 02:49:40</td>
    </tr>
    <tr>
      <th>South Beach</th>
      <th>Human or Animal Waste</th>
      <td>9</td>
      <td>3 days 00:37:45.888888</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">South of Market</th>
      <th>Human or Animal Waste</th>
      <td>309</td>
      <td>11 days 11:59:39.754045</td>
    </tr>
    <tr>
      <th>Human/Animal Waste - BART</th>
      <td>1</td>
      <td>0 days 00:05:00</td>
    </tr>
    <tr>
      <th>St. Mary's Park</th>
      <th>Human or Animal Waste</th>
      <td>1</td>
      <td>0 days 05:05:37</td>
    </tr>
    <tr>
      <th>Stonestown</th>
      <th>Human or Animal Waste</th>
      <td>1</td>
      <td>0 days 09:14:08</td>
    </tr>
    <tr>
      <th>Sunnyside</th>
      <th>Human or Animal Waste</th>
      <td>4</td>
      <td>4 days 13:42:49.500000</td>
    </tr>
    <tr>
      <th>Sutro Heights</th>
      <th>Human or Animal Waste</th>
      <td>2</td>
      <td>2 days 07:42:50</td>
    </tr>
    <tr>
      <th>Telegraph Hill</th>
      <th>Human or Animal Waste</th>
      <td>4</td>
      <td>11 days 20:27:32.250000</td>
    </tr>
    <tr>
      <th>Tenderloin</th>
      <th>Human or Animal Waste</th>
      <td>328</td>
      <td>7 days 15:48:14.289634</td>
    </tr>
    <tr>
      <th>Union Street</th>
      <th>Human or Animal Waste</th>
      <td>2</td>
      <td>1 days 04:11:36.500000</td>
    </tr>
    <tr>
      <th>Upper Market</th>
      <th>Human or Animal Waste</th>
      <td>1</td>
      <td>15 days 03:03:00</td>
    </tr>
    <tr>
      <th>Visitacion Valley</th>
      <th>Human or Animal Waste</th>
      <td>4</td>
      <td>6 days 19:19:52.500000</td>
    </tr>
    <tr>
      <th>West Portal</th>
      <th>Human or Animal Waste</th>
      <td>4</td>
      <td>2 days 21:42:55.500000</td>
    </tr>
    <tr>
      <th>Western Addition</th>
      <th>Human or Animal Waste</th>
      <td>11</td>
      <td>28 days 11:17:45.454545</td>
    </tr>
    <tr>
      <th>Westwood Park</th>
      <th>Human or Animal Waste</th>
      <td>2</td>
      <td>1 days 21:34:30</td>
    </tr>
    <tr>
      <th rowspan="7" valign="top">missing</th>
      <th>Human or Animal Waste</th>
      <td>19</td>
      <td>16 days 09:03:37.631578</td>
    </tr>
    <tr>
      <th>Human/Animal Waste - BART</th>
      <td>1</td>
      <td>0 days 19:48:00</td>
    </tr>
    <tr>
      <th>human_resources - complaint</th>
      <td>1</td>
      <td>27 days 22:49:58</td>
    </tr>
    <tr>
      <th>human_resources - customer_callback</th>
      <td>2</td>
      <td>151 days 12:14:19.500000</td>
    </tr>
    <tr>
      <th>human_resources - followup_request</th>
      <td>1</td>
      <td>399 days 21:25:46</td>
    </tr>
    <tr>
      <th>human_resources - mailing_request</th>
      <td>1</td>
      <td>244 days 07:02:00</td>
    </tr>
    <tr>
      <th>human_resources - request_for_service</th>
      <td>2</td>
      <td>1053 days 07:50:11</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">

</div>
<div class="cell border-box-sizing code_cell rendered">

<div class="output_wrapper">
<div class="output">

<div class="output_area">



<div class="output_png output_subarea ">
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">

</div>
    </div>
  </div>
</body>

 

<script type="application/vnd.jupyter.widget-state+json">
</script>


</html>