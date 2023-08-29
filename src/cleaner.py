import re
import os

def main():
    dir_files = os.listdir( 'textbook_pages_OG' )
    for file in dir_files:
        # Open old file to read in from
        filepath_old = './textbook_pages_OG/'+file
        with open( filepath_old, "r+" ) as old_file:
            old_text = old_file.read()
            # Truncate new lines
            cleaned_text = re.sub( r'\n+', '\n', old_text )
            # Get rid of the giant list of links from the "Contents" side bar
            cleaned_text = re.sub( r'Faculty ResourcesTeaching.+\n', '\n', cleaned_text )
            # Get rid of all the footer stuff at the bottom of the document
            cleaned_text = re.sub( r'Previous/next navigation[\s\S]*', '', cleaned_text )
            # Get rid of all the header stuff at the top of the document
            cleaned_text = re.sub( r'^[\s\S]*\nIntro Psych', '', cleaned_text )
            # Get rid of all HTML tags. JUST the tags, not the words in between the tags.
            cleaned_text = re.sub( r'<[^>]*>', '', cleaned_text )
            # Get rid of any and all "Show Answer" since it's pointless text
            cleaned_text = re.sub( r'Show Answer', '', cleaned_text )
            # Change new line characters into two spaces each so that words and their definitions
            # are on the same line when being processed through the text splitter in database.py
            cleaned_text = re.sub( r'\n', '  ', cleaned_text )
        # Open new file with edited changes
        filepath_new = './textbook_pages_new/'+file
        with open( filepath_new, "w+" ) as new_file:
            new_file.write( cleaned_text )


if __name__ == "__main__":
    main()