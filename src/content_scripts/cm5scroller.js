// modified from: https://github.com/cqroot/joplin-outline/blob/main/src/codeMirrorScroller.js
function plugin(CodeMirror) {
  if (CodeMirror.cm6) { return; }

  CodeMirror.defineExtension('scrollToJarvisLine', function scrollToJarvisLine(lineno) {
    // temporary fix: sometimes the first coordinate is incorrect,
    // resulting in a difference about +- 10 px,
    // call the scroll function twice fixes the problem.
    this.scrollTo(null, this.charCoords({ line: lineno, ch: 0 }, 'local').top);
    this.scrollTo(null, this.charCoords({ line: lineno, ch: 0 }, 'local').top);
  });
}

module.exports = {
  default() {
    return {
      plugin,
    };
  },
};